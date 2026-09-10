"""Tests for the gpse pipeline command (convert + train chained)."""

import pytest

import gpse.train.cli as train_cli_module
from gpse.cli import main as gpse_main
from gpse.pipeline import cli as pipeline_cli
from gpse.utils.cli_display import _build_pipeline_parser, _build_root_parser


@pytest.fixture
def fake_stages(monkeypatch):
    """Replace the convert and train stages with recording fakes."""
    calls = {}

    def fake_convert(args, mode):
        calls["convert"] = (args, mode)
        return 0

    def fake_train(argv, **kwargs):
        calls["train"] = list(argv)
        return 0

    monkeypatch.setattr(pipeline_cli, "run_convert_workflow", fake_convert)
    monkeypatch.setattr(train_cli_module, "main", fake_train)
    return calls


def _pipeline_argv(prefix, *extra):
    return [
        "--vcf", "in.vcf",
        "--pheno", "pheno.tsv",
        "--out-prefix", str(prefix),
        "--target_trait", "yield",
        "--task_type", "regression",
        *extra,
    ]


def test_root_parser_accepts_pipeline_command():
    parser = _build_root_parser()
    args = parser.parse_args(["pipeline"])
    assert args.command == "pipeline"


def test_unknown_command_message_mentions_pipeline(capsys):
    with pytest.raises(SystemExit):
        gpse_main(["nonsense"])
    assert "pipeline" in capsys.readouterr().err


def test_pipeline_parser_recognizes_convert_options():
    parser = _build_pipeline_parser()
    args, extra = parser.parse_known_args([
        "--vcf", "in.vcf", "--pheno", "pheno.tsv", "--out-prefix", "out/run1",
        "--run-qc", "--maf", "0.1",
        "--target_trait", "yield", "--task_type", "regression",
    ])
    assert args.vcf == "in.vcf"
    assert args.run_qc is True
    assert args.maf == 0.1
    assert args.target_trait == "yield"
    assert args.task_type == "regression"
    assert extra == []


def test_pipeline_parser_forwards_train_options_verbatim():
    parser = _build_pipeline_parser()
    _, extra = parser.parse_known_args([
        "--vcf", "in.vcf",
        "--trials", "50", "--threads", "32",
        "--models", "rf_reg", "xgboost_reg",
        "--no-use_stacking",
    ])
    assert extra == [
        "--trials", "50", "--threads", "32",
        "--models", "rf_reg", "xgboost_reg",
        "--no-use_stacking",
    ]


def test_pipeline_parser_threads_goes_to_train_convert_threads_to_convert():
    parser = _build_pipeline_parser()
    args, extra = parser.parse_known_args(["--convert-threads", "4", "--threads", "32"])
    assert args.threads == 4
    assert extra == ["--threads", "32"]


def test_pipeline_chains_convert_then_train(tmp_path, fake_stages):
    prefix = tmp_path / "run1"
    (tmp_path / "run1_yield_genotype.parquet").write_text("geno")
    (tmp_path / "run1_yield_phenotype.parquet").write_text("pheno")

    rc = pipeline_cli.main(_pipeline_argv(prefix, "--trials", "5"))

    assert rc == 0
    convert_args, mode = fake_stages["convert"]
    assert mode == "pipeline"
    assert convert_args.vcf == "in.vcf"

    train_argv = fake_stages["train"]
    assert train_argv[train_argv.index("--geno_file") + 1] == f"{prefix}_yield_genotype.parquet"
    assert train_argv[train_argv.index("--pheno_file") + 1] == f"{prefix}_yield_phenotype.parquet"
    assert train_argv[train_argv.index("--target_trait") + 1] == "yield"
    assert train_argv[train_argv.index("--task_type") + 1] == "regression"
    assert train_argv[train_argv.index("--trials") + 1] == "5"


def test_pipeline_sanitizes_trait_name_for_output_lookup(tmp_path, fake_stages):
    prefix = tmp_path / "run2"
    (tmp_path / "run2_Plant_Height__cm__genotype.parquet").write_text("geno")
    (tmp_path / "run2_Plant_Height__cm__phenotype.parquet").write_text("pheno")

    argv = _pipeline_argv(prefix)
    argv[argv.index("yield")] = "Plant Height (cm)"
    rc = pipeline_cli.main(argv)

    assert rc == 0
    train_argv = fake_stages["train"]
    assert train_argv[train_argv.index("--geno_file") + 1] == f"{prefix}_Plant_Height__cm__genotype.parquet"
    assert train_argv[train_argv.index("--target_trait") + 1] == "Plant Height (cm)"


def test_pipeline_skip_match_uses_prefix_matrix_and_raw_phenotype(tmp_path, fake_stages):
    prefix = tmp_path / "run3"
    (tmp_path / "run3.parquet").write_text("geno")

    argv = _pipeline_argv(prefix) + ["--skip-match"]
    rc = pipeline_cli.main(argv)

    assert rc == 0
    train_argv = fake_stages["train"]
    assert train_argv[train_argv.index("--geno_file") + 1] == f"{prefix}.parquet"
    assert train_argv[train_argv.index("--pheno_file") + 1] == "pheno.tsv"


def test_pipeline_forwards_preserve_vcf_snp_ids_to_train(tmp_path, fake_stages):
    prefix = tmp_path / "run4"
    (tmp_path / "run4_yield_genotype.parquet").write_text("geno")
    (tmp_path / "run4_yield_phenotype.parquet").write_text("pheno")

    rc = pipeline_cli.main(_pipeline_argv(prefix, "--preserve-vcf-snp-ids"))

    assert rc == 0
    assert "--preserve-vcf-snp-ids" in fake_stages["train"]


def test_pipeline_aborts_when_convert_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_cli, "run_convert_workflow", lambda args, mode: 1)

    def fail_train(argv, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("train stage must not run after a failed convert stage")

    monkeypatch.setattr(train_cli_module, "main", fail_train)

    rc = pipeline_cli.main(_pipeline_argv(tmp_path / "run5"))
    assert rc == 1


def test_pipeline_errors_when_converted_trait_files_missing(tmp_path, fake_stages):
    rc = pipeline_cli.main(_pipeline_argv(tmp_path / "run6"))
    assert rc == 1
    assert "train" not in fake_stages


def test_pipeline_requires_target_trait():
    with pytest.raises(SystemExit):
        pipeline_cli.main([
            "--vcf", "in.vcf", "--pheno", "pheno.tsv", "--out-prefix", "out",
            "--task_type", "regression",
        ])


def test_pipeline_requires_task_type():
    with pytest.raises(SystemExit):
        pipeline_cli.main([
            "--vcf", "in.vcf", "--pheno", "pheno.tsv", "--out-prefix", "out",
            "--target_trait", "yield",
        ])


def test_pipeline_rejects_conflicting_train_options():
    with pytest.raises(SystemExit):
        pipeline_cli.main([
            "--vcf", "in.vcf", "--pheno", "pheno.tsv", "--out-prefix", "out",
            "--target_trait", "yield", "--task_type", "regression",
            "--geno_file", "other.csv",
        ])
