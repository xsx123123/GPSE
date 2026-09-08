#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the GPSE MCP server module."""

import json
import sys
import time

import pytest

from gpse.mcp import server


@pytest.fixture()
def jobs_dir(tmp_path, monkeypatch):
    """Redirect the background-jobs root to a temporary directory."""
    monkeypatch.setenv("GPSE_MCP_JOBS_DIR", str(tmp_path))
    return tmp_path


def _wait_for_terminal_status(job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        meta = server._job_status(job_id)
        if meta["status"] != "running":
            return meta
        time.sleep(0.2)
    pytest.fail(f"Job {job_id} did not finish within {timeout} seconds")


def test_server_instance():
    assert server.mcp.name == "gpse"


def test_run_sync_version():
    result = server._version()
    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["stdout"].strip()


def test_run_sync_help_exit_zero():
    # ``gpse --version`` currently requires train args (pre-existing); use
    # the root help as the quick exit-0 command through the gpse CLI.
    result = server._run_sync(["--help"], timeout=120)
    assert result["ok"] is True
    assert result["exit_code"] == 0


def test_run_sync_failure_is_readable():
    result = server._run_sync(["no-such-command"], timeout=120)
    assert result["ok"] is False
    assert result["exit_code"] != 0


def test_truncate_output_keeps_tail():
    text = "x" * 9000 + "tail"
    truncated = server._truncate_output(text, max_chars=100)
    assert "truncated" in truncated
    assert truncated.endswith("tail")
    assert server._truncate_output("short") == "short"


def test_build_train_args():
    args = server._build_train_args(
        task_type="regression",
        geno_file="/data/geno.parquet",
        pheno_file="/data/pheno.csv",
        target_trait="FT",
        results_dir="/data/out",
        models=["rf", "xgboost"],
        trials=10,
        n_jobs=4,
        extra_args=["--use_default_params"],
    )
    assert args[:2] == ["train", "--task_type"]
    assert "--geno_file" in args and "/data/geno.parquet" in args
    assert args[args.index("--models") + 1 : args.index("--models") + 3] == ["rf", "xgboost"]
    assert args[-1] == "--use_default_params"
    # Empty optional values are omitted.
    assert "--n_classes" not in args


def test_build_predict_args_requires_values():
    args = server._build_predict_args(model="m.pkl", geno_file="g.vcf", out="pred.csv")
    assert args == ["predict", "--model", "m.pkl", "--geno-file", "g.vcf", "--out", "pred.csv"]


def test_job_finished_lifecycle(jobs_dir):
    meta = server._start_job(server._gpse_cmd(["--help"]))
    job_id = meta["job_id"]
    assert meta["status"] == "running"
    assert (jobs_dir / job_id / "meta.json").is_file()

    final = _wait_for_terminal_status(job_id)
    assert final["status"] == "finished"
    assert final["exit_code"] == 0

    # Terminal status is persisted in meta.json.
    persisted = json.loads((jobs_dir / job_id / "meta.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "finished"

    log = server._job_log(job_id)
    assert isinstance(log, str) and log.strip()

    listed = {job["job_id"]: job for job in server._job_list()}
    assert job_id in listed
    assert listed[job_id]["status"] == "finished"


def test_job_running_and_stop(jobs_dir):
    meta = server._start_job([sys.executable, "-c", "import time; time.sleep(300)"])
    job_id = meta["job_id"]

    status = server._job_status(job_id)
    assert status["status"] == "running"

    result = server._job_stop(job_id)
    assert result["ok"] is True

    final = _wait_for_terminal_status(job_id, timeout=15)
    assert final["status"] in {"stopped", "failed"}


def test_job_invalid_id(jobs_dir):
    assert "error" in server._job_status("not-a-job")
    assert "error" in server._job_status("../../etc")
    assert "Unknown job_id" in server._job_status("deadbeef")["error"]
    assert server._job_stop("deadbeef")["ok"] is False
    assert "Unknown job_id" in server._job_log("deadbeef")
