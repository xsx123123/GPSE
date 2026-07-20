#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the gpse batch YAML runner."""

import pytest
import yaml

from gpse.batch.runner import (
    _train_option_actions,
    build_trait_argv,
    load_batch_config,
    run_batch,
)


@pytest.fixture(scope="module")
def actions():
    return _train_option_actions()


def _write_config(tmp_path, data):
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def _base_config(**overrides):
    data = {
        "defaults": {
            "geno_file": "geno.csv",
            "pheno_file": "pheno.csv",
            "task_type": "regression",
            "use_default_params": True,
            "use_stacking": False,
            "results_root": "out",
        },
        "traits": [{"name": "FT"}, {"name": "FW"}],
    }
    data.update(overrides)
    return data


def test_load_batch_config(tmp_path):
    config_path = _write_config(tmp_path, _base_config())
    defaults, traits = load_batch_config(config_path)
    assert defaults["task_type"] == "regression"
    assert [t["name"] for t in traits] == ["FT", "FW"]


def test_load_batch_config_requires_traits(tmp_path):
    config_path = _write_config(tmp_path, {"defaults": {}})
    with pytest.raises(ValueError, match="traits"):
        load_batch_config(config_path)


def test_load_batch_config_requires_name(tmp_path):
    config_path = _write_config(tmp_path, {"traits": [{"task_type": "regression"}]})
    with pytest.raises(ValueError, match="name"):
        load_batch_config(config_path)


def test_load_batch_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_batch_config(tmp_path / "nope.yaml")


def test_build_trait_argv_inherits_defaults(actions):
    defaults, traits = _base_config()["defaults"], _base_config()["traits"]
    argv = build_trait_argv(defaults, traits[0], actions)
    assert "--target_trait" in argv
    assert argv[argv.index("--target_trait") + 1] == "FT"
    assert argv[argv.index("--task_type") + 1] == "regression"
    # store_true flags appear only when true
    assert "--use_default_params" in argv
    assert "--use_stacking" not in argv
    # results_dir derived from results_root + name
    assert argv[argv.index("--results_dir") + 1] == "out/FT"


def test_build_trait_argv_trait_overrides_and_lists(actions):
    defaults = _base_config()["defaults"]
    trait = {
        "name": "color",
        "task_type": "classification",
        "n_classes": 3,
        "models": ["rf_clf", "xgboost_clf"],
        "results_dir": "custom_out",
    }
    argv = build_trait_argv(defaults, trait, actions)
    assert argv[argv.index("--task_type") + 1] == "classification"
    assert argv[argv.index("--n_classes") + 1] == "3"
    models_at = argv.index("--models")
    assert argv[models_at + 1 : models_at + 3] == ["rf_clf", "xgboost_clf"]
    assert argv[argv.index("--results_dir") + 1] == "custom_out"


def test_build_trait_argv_rejects_unknown_option(actions):
    with pytest.raises(ValueError, match="unknown option"):
        build_trait_argv({"not_a_flag": 1, "results_root": "out"}, {"name": "FT"}, actions)


def test_build_trait_argv_rejects_explicit_target_trait(actions):
    with pytest.raises(ValueError, match="target_trait"):
        build_trait_argv({"results_root": "out"}, {"name": "FT", "target_trait": "XX"}, actions)


def test_build_trait_argv_requires_output_location(actions):
    with pytest.raises(ValueError, match="results_dir"):
        build_trait_argv({}, {"name": "FT"}, actions)


def test_build_trait_argv_boolean_optional_action(actions):
    defaults = {"results_root": "out", "use_same_test_set": True, "task_type": "regression"}
    argv = build_trait_argv(defaults, {"name": "FT"}, actions)
    assert "--use_same_test_set" in argv
    assert "True" not in argv
    argv = build_trait_argv(
        {**defaults, "use_same_test_set": False}, {"name": "FT"}, actions
    )
    assert "--no-use_same_test_set" in argv
    assert "--use_same_test_set" not in argv


def test_run_batch_dry_run(tmp_path, capsys):
    config_path = _write_config(tmp_path, _base_config())
    assert run_batch(config_path, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "gpse train" in out
    assert "--target_trait FT" in out
    assert "--target_trait FW" in out


def test_run_batch_skips_disabled(tmp_path, capsys):
    data = _base_config()
    data["traits"].append({"name": "DTF", "enabled": False})
    config_path = _write_config(tmp_path, data)
    assert run_batch(config_path, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "--target_trait DTF" not in out


def test_run_batch_continues_after_failure(tmp_path, monkeypatch):
    calls = []

    def fake_train_main(argv, prog=None):
        calls.append(argv[argv.index("--target_trait") + 1])
        return 1 if calls[-1] == "FT" else 0

    monkeypatch.setattr("gpse.train.cli.main", fake_train_main)
    config_path = _write_config(tmp_path, _base_config())
    assert run_batch(config_path) == 1
    assert calls == ["FT", "FW"]  # failure of FT did not stop FW


def test_run_batch_aborts_on_interrupt(tmp_path, monkeypatch):
    monkeypatch.setattr("gpse.train.cli.main", lambda argv, prog=None: 130)
    config_path = _write_config(tmp_path, _base_config())
    assert run_batch(config_path) == 130


def test_batch_cli_routes(monkeypatch, tmp_path):
    from gpse.batch.cli import main as batch_main

    monkeypatch.setattr(
        "gpse.batch.runner.run_batch", lambda config, dry_run=False: 0
    )
    config_path = _write_config(tmp_path, _base_config())
    assert batch_main(["--config", str(config_path)]) == 0
