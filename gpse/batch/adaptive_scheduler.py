#!/usr/bin/env python3
"""
Adaptive batch scheduler for GPSE trait training.

Features:
- Memory-aware concurrency: reads /proc/meminfo (+ cgroup v2 memory events)
  and caps concurrent traits at min(cpu_budget, mem_budget, hard_cap).
- Queue-style execution with per-task stdout/stderr log, PID file, start/end
  timestamps, and exit code.
- Resumable: re-running the same YAML skips traits whose
  ``reports/run_summary.json`` already reports a complete run.
- Auto-downgrade: on OOM-style failures it first reduces concurrency, then
  (when already at 1) reduces ``--threads``/``--max_workers`` per task until
  ``--min-threads-per-trait`` is reached.
- Failure classification: memory-related exits (OOM, Killed, MemoryError,
  BrokenProcessPool) get up to 2 retries; other failures are
  ``FAILED_NON_MEMORY`` and are not retried.
- ``--dry-run`` prints the planned schedule and computed safe concurrency
  without launching anything.
- Writes ``scheduler_status.tsv`` + ``scheduler.log`` to the results root.

Usage (see also ``launch_lowdim_adaptive.sh``):

    python -m gpse.batch.adaptive_scheduler \\
        --config batch/batch_lowdim_core.yaml \\
        --max-threads 90 --min-free-mem-gb 64 \\
        --mem-per-trait-gb 32 --threads-per-trait 8 \\
        --min-threads-per-trait 2 --max-concurrent-traits 2 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Resource detection
# --------------------------------------------------------------------------- #

MEMINFO_PATH = Path("/proc/meminfo")
CGROUP_MEM_MAX = Path("/sys/fs/cgroup/memory.max")
CGROUP_MEM_CURRENT = Path("/sys/fs/cgroup/memory.current")
CGROUP_MEM_EVENTS = Path("/sys/fs/cgroup/memory.events")

GIB = 1024 ** 3


def _read_meminfo_mem_available_bytes() -> Optional[int]:
    """Return MemAvailable in bytes from /proc/meminfo, or None."""
    try:
        text = MEMINFO_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                kb = int(parts[1])
                return kb * 1024
    return None


def _read_cgroup_v2_available_bytes() -> Optional[int]:
    """Best-effort cgroup v2 available-memory estimate (memory.max - memory.current)."""
    try:
        if not CGROUP_MEM_MAX.exists() or not CGROUP_MEM_CURRENT.exists():
            return None
        max_raw = CGROUP_MEM_MAX.read_text().strip()
        if max_raw == "max":
            return None
        current = int(CGROUP_MEM_CURRENT.read_text().strip())
        return max(0, int(max_raw) - current)
    except (OSError, ValueError):
        return None


def available_memory_bytes() -> int:
    """Return the best estimate of currently available memory in bytes.

    Priority: /proc/meminfo MemAvailable; fallback to cgroup v2 estimate.
    Raises RuntimeError if neither source is readable (rare).
    """
    val = _read_meminfo_mem_available_bytes()
    if val is not None:
        return val
    val = _read_cgroup_v2_available_bytes()
    if val is not None:
        return val
    raise RuntimeError("cannot read available memory from /proc/meminfo or cgroup v2")


def read_cgroup_oom_counts() -> Dict[str, int]:
    """Return ``{'oom': n, 'oom_kill': m}`` from cgroup v2 ``memory.events``.

    Missing file / keys return 0. Used as a monotonic counter: the scheduler
    snapshots the value before a task starts and compares after it exits.
    """
    out = {"oom": 0, "oom_kill": 0}
    try:
        if not CGROUP_MEM_EVENTS.exists():
            return out
        for line in CGROUP_MEM_EVENTS.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in out:
                out[parts[0]] = int(parts[1])
    except OSError:
        pass
    return out


def cpu_count() -> int:
    """Portable logical CPU count."""
    return os.cpu_count() or 1


# --------------------------------------------------------------------------- #
# Concurrency budget
# --------------------------------------------------------------------------- #

def compute_concurrency(
    max_threads: int,
    threads_per_trait: int,
    min_free_mem_gb: float,
    mem_per_trait_gb: float,
    max_concurrent_traits: Optional[int],
    available_bytes: Optional[int] = None,
) -> int:
    """Return the safe number of concurrently running traits.

    The result is the minimum of:
      1) floor(max_threads / threads_per_trait)
      2) floor((available_bytes - min_free_mem) / mem_per_trait)
      3) max_concurrent_traits (if set)

    If the available memory is below ``min_free_mem_gb`` the result is 0
    (caller must wait rather than launch).
    """
    if threads_per_trait <= 0:
        raise ValueError("threads_per_trait must be > 0")
    if mem_per_trait_gb <= 0:
        raise ValueError("mem_per_trait_gb must be > 0")

    by_cpu = max_threads // threads_per_trait
    if available_bytes is None:
        available_bytes = available_memory_bytes()
    free_for_tasks = available_bytes - int(min_free_mem_gb * GIB)
    by_mem = max(0, int(free_for_tasks // int(mem_per_trait_gb * GIB)))

    n = min(by_cpu, by_mem)
    if max_concurrent_traits is not None:
        n = min(n, max_concurrent_traits)
    return max(0, n)


# --------------------------------------------------------------------------- #
# Task model
# --------------------------------------------------------------------------- #

MEMORY_ERROR_KEYWORDS = (
    "out of memory",
    "out-of-memory",
    "oom",
    "killed",
    "memoryerror",
    "memory error",
    "brokenprocesspool",
    "cannot allocate memory",
)


@dataclass
class TraitTask:
    species: str
    trait: str
    command: List[str]
    results_dir: Path
    log_file: Path
    pid_file: Optional[Path] = None
    threads: int = 0
    max_workers: int = 0
    expected_models: Optional[List[str]] = None
    expected_repeats: int = 50
    enabled: bool = True

    # Runtime state
    status: str = "PENDING"      # PENDING | RUNNING | DONE | FAILED_MEM | FAILED_NON_MEMORY
    attempt: int = 0
    exit_code: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    mem_available_gb_at_start: Optional[float] = None
    error_category: str = ""

    @property
    def key(self) -> str:
        return f"{self.species}::{self.trait}"


# --------------------------------------------------------------------------- #
# Status file (JSON) — resume + auditing
# --------------------------------------------------------------------------- #

class StatusStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self._data.get(key)

    def put(self, key: str, payload: Dict[str, Any]) -> None:
        with self.lock:
            self._data[key] = payload
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, default=str), encoding="utf-8")
            tmp.replace(self.path)

    def all(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return dict(self._data)


# --------------------------------------------------------------------------- #
# Result verification
# --------------------------------------------------------------------------- #

def _is_run_complete(task: TraitTask) -> bool:
    """A run is complete only when ``reports/run_summary.json`` exists and
    reports all expected models done with ``expected_repeats`` repeats each.
    """
    summary = Path(task.results_dir) / "reports" / "run_summary.json"
    if not summary.exists():
        return False
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    # Heuristic: any top-level "status" field equal to "success" or "complete".
    status = str(data.get("status", "")).lower()
    if status in {"success", "complete", "ok", "finished"}:
        return True

    # Fallback: every expected model reports n_repeats >= expected_repeats.
    if task.expected_models:
        models_block = data.get("models") or data.get("model_results") or {}
        if not isinstance(models_block, dict):
            return False
        for m in task.expected_models:
            entry = models_block.get(m) or {}
            n = entry.get("n_repeats") or entry.get("completed_repeats") or 0
            if int(n) < task.expected_repeats:
                return False
        return True
    return True


def _classify_failure(task: TraitTask) -> str:
    """Return 'MEM' if the task log shows memory-related failure keywords."""
    try:
        text = task.log_file.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        text = ""
    if any(k in text for k in MEMORY_ERROR_KEYWORDS):
        return "MEM"
    return "OTHER"


# --------------------------------------------------------------------------- #
# Task runner
# --------------------------------------------------------------------------- #

class TaskRunner:
    def __init__(self, logger):
        self.logger = logger
        self._active: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def start(self, task: TraitTask) -> None:
        task.log_file.parent.mkdir(parents=True, exist_ok=True)
        task.results_dir.mkdir(parents=True, exist_ok=True)
        log_f = open(task.log_file, "ab", buffering=0)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(
            task.command,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        with self._lock:
            self._active[task.key] = proc
        task.status = "RUNNING"
        task.start_time = _now_iso()
        task.attempt += 1
        try:
            mem_b = available_memory_bytes()
            task.mem_available_gb_at_start = round(mem_b / GIB, 2)
        except RuntimeError:
            task.mem_available_gb_at_start = None
        pid_file = task.pid_file or (task.results_dir / "scheduler.pid")
        try:
            pid_file.write_text(str(proc.pid), encoding="utf-8")
        except OSError:
            pass
        self.logger.info(
            f"START  {task.key}  pid={proc.pid}  threads={task.threads} "
            f"max_workers={task.max_workers}  mem_avail_gb={task.mem_available_gb_at_start} "
            f"log={task.log_file}"
        )

    def poll(self, key: str) -> Optional[int]:
        with self._lock:
            proc = self._active.get(key)
        if proc is None:
            return None
        rc = proc.poll()
        return rc

    def wait(self, key: str, timeout: Optional[float] = None) -> int:
        with self._lock:
            proc = self._active.get(key)
        if proc is None:
            raise KeyError(key)
        return proc.wait(timeout=timeout)

    def terminate(self, key: str) -> None:
        with self._lock:
            proc = self._active.pop(key, None)
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    def shutdown_all(self) -> None:
        with self._lock:
            keys = list(self._active.keys())
        for k in keys:
            self.terminate(k)


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #

class AdaptiveScheduler:
    def __init__(
        self,
        tasks: List[TraitTask],
        *,
        max_threads: int,
        threads_per_trait: int,
        min_threads_per_trait: int,
        min_free_mem_gb: float,
        mem_per_trait_gb: float,
        max_concurrent_traits: Optional[int],
        max_mem_retries: int = 2,
        memory_poll_seconds: float = 15.0,
        upgrade_cooldown_seconds: float = 120.0,
        status_path: Path = Path("scheduler_status.json"),
        tsv_path: Path = Path("scheduler_status.tsv"),
        log_path: Path = Path("scheduler.log"),
        dry_run: bool = False,
    ):
        self.tasks = {t.key: t for t in tasks if t.enabled}
        self.max_threads = max_threads
        self.threads_per_trait = threads_per_trait
        self.min_threads_per_trait = min_threads_per_trait
        self.min_free_mem_gb = min_free_mem_gb
        self.mem_per_trait_gb = mem_per_trait_gb
        self.max_concurrent_traits = max_concurrent_traits
        self.max_mem_retries = max_mem_retries
        self.memory_poll_seconds = memory_poll_seconds
        self.upgrade_cooldown_seconds = upgrade_cooldown_seconds
        self.status_path = status_path
        self.tsv_path = tsv_path
        self.log_path = log_path
        self.dry_run = dry_run

        self.store = StatusStore(status_path)
        self.runner = TaskRunner(self._make_logger())
        self._stop = threading.Event()
        self._current_concurrency = compute_concurrency(
            max_threads, threads_per_trait,
            min_free_mem_gb, mem_per_trait_gb, max_concurrent_traits,
        )
        self._current_threads_per_trait = threads_per_trait
        self._last_downgrade_ts = 0.0
        self._last_upgrade_ts = 0.0
        self._last_oom_counts = read_cgroup_oom_counts()

    # --- logging ----------------------------------------------------------
    def _make_logger(self):
        import logging

        logger = logging.getLogger("gpse.adaptive_scheduler")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%dT%H:%M:%S"
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        if not self.dry_run:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(self.log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        return logger

    # --- scheduling helpers ----------------------------------------------
    def _pending(self) -> List[TraitTask]:
        return [t for t in self.tasks.values() if t.status in {"PENDING"}]

    def _running_keys(self) -> List[str]:
        return [k for k, t in self.tasks.items() if t.status == "RUNNING"]

    def _restore_from_status(self) -> None:
        """Skip tasks whose previous run was already successful."""
        for t in self.tasks.values():
            prev = self.store.get(t.key)
            if prev and prev.get("status") == "DONE" and _is_run_complete(t):
                t.status = "DONE"
                self.runner.logger.info(f"SKIP   {t.key}  (previous run already complete)")

    def _persist(self, task: TraitTask) -> None:
        payload = {
            "species": task.species,
            "trait": task.trait,
            "status": task.status,
            "attempt": task.attempt,
            "exit_code": task.exit_code,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "threads": task.threads,
            "max_workers": task.max_workers,
            "mem_available_gb_at_start": task.mem_available_gb_at_start,
            "error_category": task.error_category,
            "log_file": str(task.log_file),
            "result_dir": str(task.results_dir),
        }
        self.store.put(task.key, payload)

    # --- downgrade / upgrade ---------------------------------------------
    def _downgrade(self, reason: str) -> None:
        now = time.monotonic()
        if self._current_concurrency > 1:
            old = self._current_concurrency
            self._current_concurrency -= 1
            self.runner.logger.warning(
                f"MEMORY_PRESSURE: concurrency {old} -> {self._current_concurrency}; "
                f"threads_per_trait remains {self._current_threads_per_trait}; "
                f"reason={reason}; MemAvailable="
                f"{available_memory_bytes() / GIB:.2f} GiB"
            )
        elif self._current_threads_per_trait > self.min_threads_per_trait:
            old = self._current_threads_per_trait
            self._current_threads_per_trait = max(
                self.min_threads_per_trait, self._current_threads_per_trait // 2
            )
            self.runner.logger.warning(
                f"MEMORY_PRESSURE: concurrency stays at 1; "
                f"threads_per_trait {old} -> {self._current_threads_per_trait}; "
                f"reason={reason}"
            )
        else:
            self.runner.logger.error(
                f"MEMORY_PRESSURE: already at minimum "
                f"(concurrency=1, threads_per_trait={self.min_threads_per_trait}); "
                f"cannot downgrade further. reason={reason}"
            )
        self._last_downgrade_ts = now

    def _maybe_upgrade(self) -> None:
        now = time.monotonic()
        if now - self._last_downgrade_ts < self.upgrade_cooldown_seconds:
            return
        if now - self._last_upgrade_ts < self.upgrade_cooldown_seconds:
            return
        target = compute_concurrency(
            self.max_threads,
            self._current_threads_per_trait,
            self.min_free_mem_gb,
            self.mem_per_trait_gb,
            self.max_concurrent_traits,
        )
        if target > self._current_concurrency:
            # cautious: only bump by 1
            self._current_concurrency = min(target, self._current_concurrency + 1)
            self.runner.logger.info(
                f"MEMORY_RECOVERED: concurrency {self._current_concurrency - 1} -> "
                f"{self._current_concurrency}"
            )
            self._last_upgrade_ts = now

    # --- main loop -------------------------------------------------------
    def run(self) -> int:
        if self.dry_run:
            self._print_dry_run()
            return 0

        self._restore_from_status()
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())

        try:
            while not self._stop.is_set():
                # Finalize completed children
                for key in list(self._running_keys()):
                    rc = self.runner.poll(key)
                    if rc is None:
                        continue
                    task = self.tasks[key]
                    task.exit_code = rc
                    task.end_time = _now_iso()
                    self._finalize(task)

                # Check cgroup OOM counter increments -> pause + downgrade
                cur = read_cgroup_oom_counts()
                if (cur["oom_kill"] > self._last_oom_counts["oom_kill"]
                        or cur["oom"] > self._last_oom_counts["oom"]):
                    self._downgrade(
                        f"cgroup oom {self._last_oom_counts} -> {cur}"
                    )
                    self._last_oom_counts = cur

                # Maybe launch new tasks if budget allows
                self._maybe_upgrade()
                budget_slots = self._current_concurrency - len(self._running_keys())
                if budget_slots > 0:
                    try:
                        avail = available_memory_bytes()
                    except RuntimeError:
                        avail = 0
                    if avail < int(self.min_free_mem_gb * GIB):
                        self.runner.logger.info(
                            f"WAIT   MemAvailable {avail / GIB:.2f} GiB < "
                            f"min-free {self.min_free_mem_gb} GiB; sleeping "
                            f"{self.memory_poll_seconds}s"
                        )
                        self._stop.wait(self.memory_poll_seconds)
                        continue
                    for task in self._pending()[:budget_slots]:
                        self._apply_runtime_params(task)
                        self.runner.start(task)
                        self._persist(task)

                # Exit when nothing pending and nothing running
                if not self._pending() and not self._running_keys():
                    break

                self._stop.wait(self.memory_poll_seconds)
        finally:
            self.runner.shutdown_all()
            self._write_tsv()

        mem_fail = [t for t in self.tasks.values() if t.status == "FAILED_MEM"]
        non_mem = [t for t in self.tasks.values() if t.status == "FAILED_NON_MEMORY"]
        self.runner.logger.info(
            f"DONE  total={len(self.tasks)} done={sum(1 for t in self.tasks.values() if t.status == 'DONE')} "
            f"failed_mem={len(mem_fail)} failed_non_mem={len(non_mem)}"
        )
        return 0 if not (mem_fail or non_mem) else 1

    def _apply_runtime_params(self, task: TraitTask) -> None:
        """Inject current threads/max_workers budget into the task command."""
        task.threads = self._current_threads_per_trait
        # Derive a conservative max_workers (models-level parallelism) from the
        # per-trait thread budget, keeping n_jobs=1 and repeat_workers=1 so the
        # total process count stays at threads_per_trait.
        task.max_workers = max(1, self._current_threads_per_trait)
        # Patch the command in place: replace --threads/--max_workers values.
        cmd = list(task.command)
        cmd = _upsert_flag(cmd, "--threads", str(task.threads))
        cmd = _upsert_flag(cmd, "--max_workers", str(task.max_workers))
        cmd = _upsert_flag(cmd, "--n_jobs", "1")
        cmd = _upsert_flag(cmd, "--repeat_workers", "1")
        task.command = cmd

    def _finalize(self, task: TraitTask) -> None:
        rc = task.exit_code
        if rc == 0 and _is_run_complete(task):
            task.status = "DONE"
        else:
            cat = _classify_failure(task)
            task.error_category = cat
            if cat == "MEM" and task.attempt <= self.max_mem_retries:
                self.runner.logger.warning(
                    f"RETRY  {task.key}  attempt {task.attempt}/{self.max_mem_retries} "
                    f"(exit={rc}); downgrading before relaunch"
                )
                self._downgrade(f"{task.key} OOM exit={rc}")
                task.status = "PENDING"
                task.exit_code = None
                task.start_time = None
                task.end_time = None
            elif cat == "MEM":
                task.status = "FAILED_MEM"
            else:
                task.status = "FAILED_NON_MEMORY"
        self._persist(task)

    # --- reporting -------------------------------------------------------
    def _print_dry_run(self) -> None:
        log = self.runner.logger
        log.info("=== DRY RUN ===")
        log.info(f"max_threads={self.max_threads}")
        log.info(f"threads_per_trait={self.threads_per_trait}")
        log.info(f"min_threads_per_trait={self.min_threads_per_trait}")
        log.info(f"min_free_mem_gb={self.min_free_mem_gb}")
        log.info(f"mem_per_trait_gb={self.mem_per_trait_gb}")
        log.info(f"max_concurrent_traits={self.max_concurrent_traits}")
        try:
            avail = available_memory_bytes()
            log.info(f"MemAvailable={avail / GIB:.2f} GiB")
        except RuntimeError:
            log.info("MemAvailable=<unreadable>")
            avail = None
        n = compute_concurrency(
            self.max_threads, self.threads_per_trait,
            self.min_free_mem_gb, self.mem_per_trait_gb,
            self.max_concurrent_traits, avail,
        )
        log.info(f"computed_safe_concurrency={n}")
        log.info(f"oom_counters={read_cgroup_oom_counts()}")
        log.info(f"tasks={len(self.tasks)}")
        for t in self.tasks.values():
            log.info(
                f"  - {t.key}: results_dir={t.results_dir}  "
                f"cmd={' '.join(t.command)}"
            )

    def _write_tsv(self) -> None:
        cols = [
            "species", "trait", "status", "attempt", "exit_code",
            "start_time", "end_time", "threads", "max_workers",
            "mem_available_gb", "error_category", "log_file", "result_dir",
        ]
        self.tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tsv_path, "w", encoding="utf-8") as fh:
            fh.write("\t".join(cols) + "\n")
            for t in self.tasks.values():
                row = [
                    t.species, t.trait, t.status, str(t.attempt),
                    str(t.exit_code) if t.exit_code is not None else "",
                    t.start_time or "", t.end_time or "",
                    str(t.threads), str(t.max_workers),
                    str(t.mem_available_gb_at_start) if t.mem_available_gb_at_start is not None else "",
                    t.error_category,
                    str(t.log_file), str(t.results_dir),
                ]
                fh.write("\t".join(row) + "\n")
        self.runner.logger.info(f"WROTE  {self.tsv_path}")


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #

def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _upsert_flag(cmd: List[str], flag: str, value: str) -> List[str]:
    """Insert or replace ``flag value`` in a command argv list."""
    out = list(cmd)
    if flag in out:
        i = out.index(flag)
        if i + 1 < len(out):
            out[i + 1] = value
        else:
            out.append(value)
    else:
        out.extend([flag, value])
    return out


def _load_config(path: Path) -> Dict[str, Any]:
    import yaml  # lazy: PyYAML is a project dep

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _build_tasks_from_config(cfg: Dict[str, Any], results_root: Path) -> List[TraitTask]:
    """Shape::

        defaults:
          geno_file: ...
          pheno_file: ...
          task_type: regression
          n_repeats: 50
          n_splits: 5
          cv_file: null
          use_stacking: true
          top_n_models: 3
          random_seed: 42
          models: null           # list[str] or null -> all
          expected_models: null  # used for run_summary verification
          threads: 8
          max_workers: 8
          gpse_command: ["python", "-m", "gpse", "train"]
        traits:
          - {species: tomato, name: FW}
          - {species: tomato, name: FT, cv_file: /x/FT.csv}
    """
    defs = cfg.get("defaults", {}) or {}
    traits = cfg.get("traits", []) or []
    tasks: List[TraitTask] = []
    for tr in traits:
        if not tr.get("enabled", True):
            continue
        species = tr.get("species") or defs.get("species") or "species"
        name = tr["name"]
        merged = {**defs, **tr}
        res_dir = Path(merged.get("results_dir") or (results_root / name))
        log_dir = res_dir / "scheduler_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}.log"

        cmd = list(merged.get("gpse_command") or ["python", "-m", "gpse", "train"])
        cmd += [
            "--geno_file", str(merged["geno_file"]),
            "--pheno_file", str(merged["pheno_file"]),
            "--target_trait", name,
            "--task_type", str(merged.get("task_type", "regression")),
            "--results_dir", str(res_dir),
            "--n_repeats", str(merged.get("n_repeats", 50)),
            "--n_splits", str(merged.get("n_splits", 5)),
            "--random_seed", str(merged.get("random_seed", 42)),
            "--threads", str(merged.get("threads", 8)),
            "--max_workers", str(merged.get("max_workers", 8)),
            "--n_jobs", "1",
            "--repeat_workers", "1",
        ]
        if merged.get("cv_file"):
            cmd += ["--cv_file", str(merged["cv_file"])]
        if merged.get("use_stacking"):
            cmd.append("--use_stacking")
            if merged.get("top_n_models"):
                cmd += ["--top_n_models", str(merged["top_n_models"])]
        if merged.get("models"):
            cmd += ["--models"] + [str(m) for m in merged["models"]]

        tasks.append(TraitTask(
            species=str(species),
            trait=name,
            command=cmd,
            results_dir=res_dir,
            log_file=log_file,
            threads=int(merged.get("threads", 8)),
            max_workers=int(merged.get("max_workers", 8)),
            expected_models=merged.get("expected_models"),
            expected_repeats=int(merged.get("n_repeats", 50)),
        ))
    return tasks


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gpse-adaptive-scheduler",
        description="Memory-aware adaptive batch scheduler for GPSE trait training.",
    )
    p.add_argument("--config", type=Path, required=True,
                   help="YAML describing traits and GPSE command.")
    p.add_argument("--results-root", type=Path, default=None,
                   help="Override results root (default: <config>/results).")
    p.add_argument("--max-threads", type=int, default=90)
    p.add_argument("--min-free-mem-gb", type=float, default=64.0)
    p.add_argument("--mem-per-trait-gb", type=float, default=32.0)
    p.add_argument("--threads-per-trait", type=int, default=8)
    p.add_argument("--min-threads-per-trait", type=int, default=2)
    p.add_argument("--max-concurrent-traits", type=int, default=None)
    p.add_argument("--max-mem-retries", type=int, default=2)
    p.add_argument("--memory-poll-seconds", type=float, default=15.0)
    p.add_argument("--upgrade-cooldown-seconds", type=float, default=120.0)
    p.add_argument("--status-file", type=Path, default=None)
    p.add_argument("--tsv-file", type=Path, default=None)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    cfg = _load_config(args.config)
    results_root = args.results_root or Path(args.config).parent / "results"
    tasks = _build_tasks_from_config(cfg, results_root)
    if not tasks:
        print("No enabled traits found in config.", file=sys.stderr)
        return 2

    status_path = args.status_file or (results_root / "scheduler_status.json")
    tsv_path = args.tsv_file or (results_root / "scheduler_status.tsv")
    log_path = args.log_file or (results_root / "scheduler.log")
    results_root.mkdir(parents=True, exist_ok=True)

    sched = AdaptiveScheduler(
        tasks,
        max_threads=args.max_threads,
        threads_per_trait=args.threads_per_trait,
        min_threads_per_trait=args.min_threads_per_trait,
        min_free_mem_gb=args.min_free_mem_gb,
        mem_per_trait_gb=args.mem_per_trait_gb,
        max_concurrent_traits=args.max_concurrent_traits,
        max_mem_retries=args.max_mem_retries,
        memory_poll_seconds=args.memory_poll_seconds,
        upgrade_cooldown_seconds=args.upgrade_cooldown_seconds,
        status_path=status_path,
        tsv_path=tsv_path,
        log_path=log_path,
        dry_run=args.dry_run,
    )
    return sched.run()


if __name__ == "__main__":
    sys.exit(main())
