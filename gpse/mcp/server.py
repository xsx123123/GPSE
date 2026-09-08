#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) server for GPSE.

Exposes the GPSE genomic prediction workflows (convert / train / predict /
batch / tools) as MCP tools so AI agents can drive them over stdio.

Execution model:
- Every invocation runs as a subprocess ``[sys.executable, "-m", "gpse.cli", ...]``
  (list form, no shell), so arguments are never subject to shell injection.
- Short commands run synchronously with a timeout; long-running commands
  (train, batch) are started as background jobs whose logs and metadata live
  under ``$GPSE_MCP_JOBS_DIR`` (default ``~/.gpse/mcp_jobs/<job_id>/``).

The stdio transport requires stdout to carry pure JSON-RPC, so this module
must never print to stdout (no logo, no logging on stdout).
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gpse")

_DEFAULT_TIMEOUT = 300
_MAX_OUTPUT_CHARS = 8000
_JOB_ID_RE = re.compile(r"^[0-9a-f]{8}$")
_VALID_COMMANDS = ("convert", "train", "predict", "batch", "tools")


# ---------------------------------------------------------------------------
# Synchronous execution
# ---------------------------------------------------------------------------


def _gpse_cmd(args: list[str]) -> list[str]:
    """Build the subprocess command for a GPSE CLI invocation."""
    return [sys.executable, "-m", "gpse.cli", *args]


def _truncate_output(text: str, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Keep the tail of long outputs, noting how much was dropped."""
    if len(text) <= max_chars:
        return text
    return (
        f"[truncated: showing last {max_chars} of {len(text)} characters]\n"
        + text[-max_chars:]
    )


def _run_cmd(
    cmd: list[str], timeout: int = _DEFAULT_TIMEOUT, display_cmd: str | None = None
) -> dict:
    """
    Run a command list synchronously and capture the result.

    Returns a dict with ok, exit_code, stdout, stderr, and cmd. Timeouts and
    OS-level failures are reported as readable errors instead of raising.
    """
    display_cmd = display_cmd or " ".join(cmd)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"Command timed out after {timeout} seconds: {display_cmd}",
            "cmd": display_cmd,
        }
    except OSError as exc:
        return {"ok": False, "error": f"Failed to start command: {exc}", "cmd": display_cmd}
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": _truncate_output(proc.stdout),
        "stderr": _truncate_output(proc.stderr),
        "cmd": display_cmd,
    }


def _run_sync(args: list[str], timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """Run ``gpse <args...>`` synchronously; see :func:`_run_cmd`."""
    return _run_cmd(_gpse_cmd(args), timeout=timeout, display_cmd="gpse " + " ".join(args))


def _version() -> dict:
    """
    Return the installed gpse package version.

    ``gpse --version`` currently routes through the train parser, which
    requires ``--task_type`` and exits non-zero, so the version is read from
    package metadata in a subprocess instead (keeps the execution model and
    works regardless of the server's own import state).
    """
    cmd = [sys.executable, "-c", "import gpse; print(gpse.__version__)"]
    return _run_cmd(cmd, timeout=60, display_cmd="python -c 'import gpse; print(gpse.__version__)'")


# ---------------------------------------------------------------------------
# Background job management
# ---------------------------------------------------------------------------


def _jobs_root() -> Path:
    """Directory that stores background job metadata and logs."""
    override = os.environ.get("GPSE_MCP_JOBS_DIR")
    return Path(override) if override else Path.home() / ".gpse" / "mcp_jobs"


def _write_meta(job_dir: Path, meta: dict) -> None:
    (job_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _start_job(cmd: list[str]) -> dict:
    """
    Start a background job for the full command list ``cmd``.

    The process runs in its own session so the whole process group can be
    stopped later. stdout and stderr are merged into ``output.log`` inside
    the job directory; metadata goes to ``meta.json``.
    """
    job_id = uuid.uuid4().hex[:8]
    job_dir = _jobs_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(job_dir / "output.log", "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise
    meta = {
        "job_id": job_id,
        "cmd": " ".join(cmd),
        "pid": proc.pid,
        "start_time": time.time(),
        "status": "running",
    }
    _write_meta(job_dir, meta)
    return meta


def _probe_pid(pid: int) -> tuple[bool, int | None]:
    """
    Check whether ``pid`` is still running.

    Returns ``(alive, exit_code)``. ``exit_code`` is only known when the
    process is our child and has just been reaped here; it is None when the
    exit status is unavailable (e.g. the server restarted since job start).
    """
    try:
        wpid, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        # Not our child (foreign pid or already reaped): fall back to a
        # liveness probe without knowing the exit code.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False, None
        except PermissionError:
            return True, None
        return True, None
    except OSError:
        return False, None
    if wpid == 0:
        return True, None
    if os.WIFEXITED(status):
        return False, os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return False, -os.WTERMSIG(status)
    return False, None


def _load_job(job_id: str) -> tuple[Path, dict] | dict:
    """
    Load and refresh a job by id.

    Returns ``(job_dir, meta)`` on success, or an error dict when the id is
    malformed or unknown.
    """
    if not _JOB_ID_RE.match(job_id or ""):
        return {"ok": False, "error": f"Invalid job_id: {job_id!r} (expected 8 hex characters)"}
    job_dir = _jobs_root() / job_id
    meta_path = job_dir / "meta.json"
    if not meta_path.is_file():
        return {"ok": False, "error": f"Unknown job_id: {job_id}"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return job_dir, _refresh_job(job_dir, meta)


def _refresh_job(job_dir: Path, meta: dict) -> dict:
    """Update the persisted status of a running job by probing its pid."""
    if meta.get("status") != "running":
        return meta
    alive, exit_code = _probe_pid(int(meta["pid"]))
    if alive:
        return meta
    if exit_code is None:
        # Process is gone but the exit code is unavailable (typically the
        # MCP server was restarted and the job is no longer our child).
        meta["status"] = "exited"
    else:
        meta["status"] = "finished" if exit_code == 0 else "failed"
        meta["exit_code"] = exit_code
    meta["end_time"] = time.time()
    _write_meta(job_dir, meta)
    return meta


def _job_status(job_id: str) -> dict:
    """Return the refreshed metadata dict for one job."""
    loaded = _load_job(job_id)
    if isinstance(loaded, dict):
        return loaded
    _, meta = loaded
    return meta


def _job_log(job_id: str, tail: int = 4000) -> str:
    """Return the last ``tail`` characters of a job's output log."""
    loaded = _load_job(job_id)
    if isinstance(loaded, dict):
        return loaded["error"]
    job_dir, _ = loaded
    log_path = job_dir / "output.log"
    if not log_path.is_file():
        return f"No output log found for job {job_id}."
    return _truncate_output(log_path.read_text(encoding="utf-8", errors="replace"), tail)


def _job_stop(job_id: str, grace_seconds: float = 5.0) -> dict:
    """
    Terminate a running job's whole process group (SIGTERM, then SIGKILL).

    Jobs that already ended are reported with their final status.
    """
    loaded = _load_job(job_id)
    if isinstance(loaded, dict):
        return loaded
    job_dir, meta = loaded
    if meta.get("status") != "running":
        return {"ok": True, "job_id": job_id, "status": meta.get("status"),
                "message": "Job is not running; nothing to stop."}
    pid = int(meta["pid"])
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        else:  # pragma: no cover - non-POSIX fallback
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.time() + grace_seconds
    exit_code = None
    while True:
        alive, exit_code = _probe_pid(pid)
        if not alive:
            break
        if time.time() >= deadline:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                else:  # pragma: no cover - non-POSIX fallback
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(0.2)
            _, exit_code = _probe_pid(pid)  # reap the child if it is ours
            break
        time.sleep(0.1)
    # A job terminated by an explicit stop request is reported as "stopped"
    # regardless of the signal exit code; the exit code is kept for reference.
    meta["status"] = "stopped"
    if exit_code is not None:
        meta["exit_code"] = exit_code
    meta["end_time"] = time.time()
    _write_meta(job_dir, meta)
    return {"ok": True, "job_id": job_id, "status": "stopped", "message": "Job stopped."}


def _job_list() -> list[dict]:
    """List all known jobs, newest first, with refreshed statuses."""
    root = _jobs_root()
    if not root.is_dir():
        return []
    jobs = []
    for job_dir in sorted(
        (d for d in root.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    ):
        meta_path = job_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        jobs.append(_refresh_job(job_dir, meta))
    return jobs


# ---------------------------------------------------------------------------
# Argument builders (plain functions, unit-testable)
# ---------------------------------------------------------------------------


def _append_opt(args: list[str], flag: str, value) -> None:
    """Append ``flag value`` when value is a non-empty, non-None option."""
    if value is None or value == "":
        return
    args.extend([flag, str(value)])


def _build_convert_args(
    vcf: str = "",
    bfile: str = "",
    ped_file: str = "",
    map_file: str = "",
    matrix_file: str = "",
    pheno: str = "",
    out_prefix: str = "",
    out_format: str = "",
    geno_encoding: str = "",
    threads: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    args = ["convert"]
    _append_opt(args, "--vcf", vcf)
    _append_opt(args, "--bfile", bfile)
    _append_opt(args, "--ped-file", ped_file)
    _append_opt(args, "--map-file", map_file)
    _append_opt(args, "--matrix-file", matrix_file)
    _append_opt(args, "--pheno", pheno)
    _append_opt(args, "--out-prefix", out_prefix)
    _append_opt(args, "--out-format", out_format)
    _append_opt(args, "--geno-encoding", geno_encoding)
    _append_opt(args, "--threads", threads)
    args.extend(extra_args or [])
    return args


def _build_train_args(
    task_type: str,
    geno_file: str = "",
    pheno_file: str = "",
    target_trait: str = "",
    results_dir: str = "",
    models: list[str] | None = None,
    trials: int | None = None,
    n_repeats: int | None = None,
    n_jobs: int | None = None,
    n_classes: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    args = ["train", "--task_type", task_type]
    _append_opt(args, "--geno_file", geno_file)
    _append_opt(args, "--pheno_file", pheno_file)
    _append_opt(args, "--target_trait", target_trait)
    _append_opt(args, "--results_dir", results_dir)
    _append_opt(args, "--trials", trials)
    _append_opt(args, "--n_repeats", n_repeats)
    _append_opt(args, "--n_jobs", n_jobs)
    _append_opt(args, "--n_classes", n_classes)
    if models:
        args.append("--models")
        args.extend(models)
    args.extend(extra_args or [])
    return args


def _build_predict_args(
    model: str,
    geno_file: str,
    out: str,
    report: str = "",
    missing_value: float | None = None,
    min_feature_coverage: float | None = None,
    preserve_vcf_snp_ids: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    args = ["predict", "--model", model, "--geno-file", geno_file, "--out", out]
    _append_opt(args, "--report", report)
    _append_opt(args, "--missing-value", missing_value)
    _append_opt(args, "--min-feature-coverage", min_feature_coverage)
    if preserve_vcf_snp_ids:
        args.append("--preserve-vcf-snp-ids")
    args.extend(extra_args or [])
    return args


def _error(message: str) -> dict:
    return {"ok": False, "error": message}


def _validate_extra_args(extra_args: list[str] | None) -> list[str]:
    return [str(a) for a in (extra_args or [])]


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
def gpse_version() -> dict:
    """
    Return the installed GPSE version.

    Use this first to verify GPSE is reachable from the MCP server. Note:
    this reads package metadata rather than running ``gpse --version``
    (which currently requires train arguments and exits non-zero).
    """
    return _version()


@mcp.tool()
def gpse_help(command: str = "") -> str:
    """
    Return the CLI help text for GPSE or one of its subcommands.

    Args:
        command: One of "convert", "train", "predict", "batch", "tools", or
            "" for the top-level help. Use this to discover the exact CLI
            flags before calling the other gpse_* tools, especially when you
            need options that are only reachable via their extra_args.
    """
    if command and command not in _VALID_COMMANDS:
        return (
            f"Unknown command: {command!r}. "
            f"Use one of: {', '.join(_VALID_COMMANDS)} (or empty for top-level help)."
        )
    args = [command, "--help"] if command else ["--help"]
    result = _run_sync(args, timeout=60)
    output = result.get("stdout", "") or result.get("stderr", "")
    if not result["ok"] and not output:
        return result.get("error", "Failed to retrieve help output.")
    return output


@mcp.tool()
def gpse_convert(
    vcf: str = "",
    bfile: str = "",
    ped_file: str = "",
    map_file: str = "",
    matrix_file: str = "",
    pheno: str = "",
    out_prefix: str = "",
    out_format: str = "",
    geno_encoding: str = "",
    threads: int | None = None,
    extra_args: list[str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict:
    """
    Run ``gpse convert``: convert raw genotype/phenotype inputs (VCF, PLINK
    bfile, PED/MAP, or an existing matrix) into training-ready numeric
    matrices, optionally with QC / LD pruning.

    Paths may be absolute or relative to the MCP server's working directory.

    Args:
        vcf: Input VCF file path (genotype source; pick exactly one source).
        bfile: PLINK BED/BIM/FAM prefix (alternative genotype source).
        ped_file: PED file path (use together with map_file).
        map_file: MAP file path (companion to ped_file).
        matrix_file: Existing genotype matrix CSV/Parquet (alternative source).
        pheno: Input phenotype file path.
        out_prefix: Output file prefix.
        out_format: Output matrix format: "csv", "parquet" (default), "feather".
        geno_encoding: "012" (default) or "-101".
        threads: Parallel threads (default 10).
        extra_args: Additional raw CLI flags passed through verbatim, e.g.
            ["--run-qc", "--maf", "0.05"]. Call gpse_help("convert") first to
            discover available flags.
        timeout: Maximum seconds to wait (default 300).
    """
    args = _build_convert_args(
        vcf=vcf, bfile=bfile, ped_file=ped_file, map_file=map_file,
        matrix_file=matrix_file, pheno=pheno, out_prefix=out_prefix,
        out_format=out_format, geno_encoding=geno_encoding, threads=threads,
        extra_args=_validate_extra_args(extra_args),
    )
    return _run_sync(args, timeout=timeout)


@mcp.tool()
def gpse_train(
    task_type: str,
    geno_file: str = "",
    pheno_file: str = "",
    target_trait: str = "",
    results_dir: str = "",
    models: list[str] | None = None,
    trials: int | None = None,
    n_repeats: int | None = None,
    n_jobs: int | None = None,
    n_classes: int | None = None,
    extra_args: list[str] | None = None,
    wait: bool = False,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict:
    """
    Run ``gpse train``: train genomic prediction models (stacking ensemble).

    Training can take a long time, so by default this starts a BACKGROUND job
    and returns a job_id immediately. Track it with gpse_job_status /
    gpse_job_log, and stop it with gpse_job_stop. Set wait=True to run
    synchronously with a timeout instead (only for quick/small runs).

    Paths may be absolute or relative to the MCP server's working directory.

    Args:
        task_type: "regression" or "classification" (required).
        geno_file: Genotype matrix file (CSV/Parquet, e.g. from gpse_convert).
        pheno_file: Phenotype file (CSV).
        target_trait: Target trait column name in the phenotype file.
        results_dir: Output directory for results (default: optimization_results_v2).
        models: Model names to train (default: config-defined set).
        trials: Optuna trials per model (default 100).
        n_repeats: Repeated training iterations (default 100).
        n_jobs: Native thread count for numerical libraries (default 1).
        n_classes: Number of classes (required for classification).
        extra_args: Additional raw CLI flags, e.g. ["--use_default_params",
            "--no-use_stacking"]. Call gpse_help("train") to discover flags.
        wait: Run synchronously instead of starting a background job.
        timeout: Seconds to wait when wait=True (default 300).
    """
    if task_type not in ("regression", "classification"):
        return _error(f"task_type must be 'regression' or 'classification', got {task_type!r}.")
    args = _build_train_args(
        task_type=task_type, geno_file=geno_file, pheno_file=pheno_file,
        target_trait=target_trait, results_dir=results_dir, models=models,
        trials=trials, n_repeats=n_repeats, n_jobs=n_jobs, n_classes=n_classes,
        extra_args=_validate_extra_args(extra_args),
    )
    if wait:
        return _run_sync(args, timeout=timeout)
    meta = _start_job(_gpse_cmd(args))
    return {
        "ok": True,
        "job_id": meta["job_id"],
        "pid": meta["pid"],
        "cmd": meta["cmd"],
        "message": "Training started in the background. Use gpse_job_status / "
                   "gpse_job_log / gpse_job_stop with this job_id.",
    }


@mcp.tool()
def gpse_predict(
    model: str,
    geno_file: str,
    out: str,
    report: str = "",
    missing_value: float | None = None,
    min_feature_coverage: float | None = None,
    preserve_vcf_snp_ids: bool = False,
    extra_args: list[str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict:
    """
    Run ``gpse predict``: predict phenotypes for new samples using a trained
    GPSE model, with canonical SNP-ID alignment between model and input.

    Paths may be absolute or relative to the MCP server's working directory.

    Args:
        model: Model artifact (.pkl) or GPSE results directory (required).
        geno_file: VCF (.vcf/.vcf.gz) or converted genotype matrix (required).
        out: Prediction CSV output file (required).
        report: Alignment report JSON path (default: <out stem>.alignment.json).
        missing_value: Fill value for model SNPs absent from input (default 3).
        min_feature_coverage: Reject when matched model-SNP coverage is below
            this 0-1 threshold (default 0).
        preserve_vcf_snp_ids: Compatibility mode; must match training mode.
        extra_args: Additional raw CLI flags. See gpse_help("predict").
        timeout: Maximum seconds to wait (default 300).
    """
    if not model or not geno_file or not out:
        return _error("model, geno_file, and out are all required.")
    args = _build_predict_args(
        model=model, geno_file=geno_file, out=out, report=report,
        missing_value=missing_value, min_feature_coverage=min_feature_coverage,
        preserve_vcf_snp_ids=preserve_vcf_snp_ids,
        extra_args=_validate_extra_args(extra_args),
    )
    return _run_sync(args, timeout=timeout)


@mcp.tool()
def gpse_batch(
    config: str,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
    wait: bool = False,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict:
    """
    Run ``gpse batch``: train GPSE models for multiple traits from one YAML
    config file (see gpse_help("batch") for the config schema).

    Like gpse_train, this starts a BACKGROUND job by default and returns a
    job_id; use wait=True for a synchronous run with a timeout.

    Args:
        config: Path to the YAML batch config (required; absolute path, or
            relative to the MCP server's working directory).
        dry_run: Only print the generated 'gpse train' commands without
            running them (fast; pair with wait=True).
        extra_args: Additional raw CLI flags.
        wait: Run synchronously instead of starting a background job.
        timeout: Seconds to wait when wait=True (default 300).
    """
    if not config:
        return _error("config is required.")
    args = ["batch", "--config", config]
    if dry_run:
        args.append("--dry_run")
    args.extend(_validate_extra_args(extra_args))
    if wait:
        return _run_sync(args, timeout=timeout)
    meta = _start_job(_gpse_cmd(args))
    return {
        "ok": True,
        "job_id": meta["job_id"],
        "pid": meta["pid"],
        "cmd": meta["cmd"],
        "message": "Batch training started in the background. Use gpse_job_status / "
                   "gpse_job_log / gpse_job_stop with this job_id.",
    }


@mcp.tool()
def gpse_tools(
    command: str,
    args: list[str] | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> dict:
    """
    Run a ``gpse tools`` utility subcommand, e.g. "split" (split genotype &
    phenotype data into train/test subsets).

    Args:
        command: Tool subcommand name, e.g. "split" (required).
        args: Arguments for the subcommand, e.g. ["--geno", "g.parquet",
            "--pheno", "p.csv", "--out-prefix", "/tmp/split"]. Call
            gpse_help("tools") to discover subcommands and their flags.
        timeout: Maximum seconds to wait (default 300).
    """
    if not command or command.startswith("-"):
        return _error(f"Invalid tools subcommand: {command!r}. Call gpse_help('tools') first.")
    return _run_sync(["tools", command, *_validate_extra_args(args)], timeout=timeout)


@mcp.tool()
def gpse_job_list() -> list[dict]:
    """
    List all background jobs started by this MCP server (newest first), with
    refreshed statuses: running / finished / failed / exited / stopped.
    """
    return _job_list()


@mcp.tool()
def gpse_job_status(job_id: str) -> dict:
    """
    Get the status of a background job.

    Args:
        job_id: The 8-character job id returned by gpse_train / gpse_batch.
    """
    return _job_status(job_id)


@mcp.tool()
def gpse_job_log(job_id: str, tail: int = 4000) -> str:
    """
    Read the tail of a background job's combined stdout/stderr log.

    Args:
        job_id: The 8-character job id returned by gpse_train / gpse_batch.
        tail: Maximum number of trailing characters to return (default 4000).
    """
    return _job_log(job_id, tail=tail)


@mcp.tool()
def gpse_job_stop(job_id: str) -> dict:
    """
    Stop a running background job by terminating its whole process group.

    Args:
        job_id: The 8-character job id returned by gpse_train / gpse_batch.
    """
    return _job_stop(job_id)


# ---------------------------------------------------------------------------
# MCP resources
# ---------------------------------------------------------------------------


@mcp.resource("gpse://config/default")
def default_config_resource() -> str:
    """Contents of the GPSE package default configuration (default.yaml)."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    try:
        return config_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"[ERROR] Cannot read default config: {exc}"


@mcp.resource("gpse://jobs/{job_id}/log")
def job_log_resource(job_id: str) -> str:
    """Combined stdout/stderr log of a background job (last 8000 chars)."""
    return _job_log(job_id, tail=_MAX_OUTPUT_CHARS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the GPSE MCP server on stdio (must not print to stdout)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
