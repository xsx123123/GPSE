# 02. Subcommand `gpse convert`

Converts VCF + phenotype data into trainable numeric matrices, and provides QC, recoding, and dependency checks. Implemented in `gpse/convert/` (CLI: `gpse/convert/cli.py`, workflow: `gpse/convert/workflow.py`).

## Four Modes

| Mode | Trigger | Function |
|------|---------|----------|
| pipeline (default) | plain invocation | VCF→PLINK→PED/MAP→numeric matrix→phenotype matching |
| qc | `--run-qc` | QC filtering + optional Beagle imputation + LD pruning |
| recode | `--recode-prefix PREFIX` | PED/MAP compound genotypes → additive numeric encoding |
| deps | `--check-deps` | Check PLINK / Java / Beagle external dependencies |

## pipeline Mode

```bash
gpse convert --vcf data.vcf.gz --pheno pheno.csv --out-prefix out/run \
  [--extract snps.txt] [--snp-dir DIR] [--direct] \
  [--skip-matrix] [--skip-match] [--skip-clean] \
  [--out-format csv|parquet|feather] [--allow-extra-chr] \
  [--trait-name yield] [--standardize-phenotype]
```

- Required: `--vcf`, `--pheno`, `--out-prefix`;
- Resume from intermediate products: `--bfile / --ped-file / --map-file / --matrix-file` continue from any intermediate step;
- `--skip-*` skips already-completed steps; `--out-format` defaults to parquet.

## qc Mode

```bash
gpse convert --run-qc --input-prefix PLINK_PREFIX \
  [--snpmaxmiss 0.1] [--samplemaxmiss 0.1] [--maf 0.05] \
  [--r2-cutoff 0.2] [--ld-window 50] [--ld-step 10] [--impute]
```

- `--snpmaxmiss` / `--samplemaxmiss`: maximum SNP / sample missing rate;
- `--maf`: minimum allele frequency; `--r2-cutoff` + `--ld-window` + `--ld-step`: LD pruning parameters;
- `--impute`: enable Beagle imputation (requires `beagle_jar_path` and java).

## recode Mode

```bash
gpse convert --recode-prefix PLINK_PREFIX
```

Converts compound genotypes in PED/MAP (e.g. `A A` / `A G`) to 0/1/2 additive numeric encoding.

## deps Mode

```bash
gpse convert --check-deps [--plink-path plink] [--java-path java] \
  [--beagle-jar-path beagle.jar] [--require-beagle]
```

## Common Options

- `-t/--threads` (default 10): number of threads;
- `--config PATH`: append an extra config file; `--no-project-config`: disable automatic project config loading;
- `-l/--log-level`: logging level.

## Corresponding API

See [06. API Reference → convert](06-api-reference.md#gpseconvert).
