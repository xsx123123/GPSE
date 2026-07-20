"""Canonical SNP identifiers shared by convert, train, and predict."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def normalize_chromosome(chromosome: object) -> str:
    """Return a stable chromosome token with a single ``chr`` prefix."""
    value = str(chromosome).strip()
    if not value:
        raise ValueError("Chromosome cannot be empty")
    return value if value.lower().startswith("chr") else f"chr{value}"


def canonical_snp_id(chromosome: object, position: int, reference: str | None = None) -> str:
    """Build ``chr<chrom>_<chromStart>_<chromEnd>`` from a VCF-style position.

    VCF positions are 1-based.  The canonical coordinates use UCSC-style
    zero-based, half-open coordinates.  For a SNP this is ``POS - 1_POS``.
    ``reference`` is accepted so the same function remains correct for a
    reference allele longer than one base.
    """
    position = int(position)
    if position < 1:
        raise ValueError(f"VCF position must be >= 1, got {position}")
    start = position - 1
    end = start + max(len(reference or ""), 1)
    return f"{normalize_chromosome(chromosome)}_{start}_{end}"


def canonical_snp_id_from_map(chromosome: object, position: object) -> str:
    """Build a canonical SNP ID from a PLINK MAP/BIM chromosome and bp."""
    return canonical_snp_id(chromosome, int(float(position)))


def ensure_unique_feature_ids(feature_ids: Iterable[str], *, source: str = "features") -> list[str]:
    """Validate and return feature IDs in their original order."""
    result = [str(feature_id).strip() for feature_id in feature_ids]
    if any(not feature_id for feature_id in result):
        raise ValueError(f"{source} contains an empty feature ID")
    duplicates = sorted({feature_id for feature_id in result if result.count(feature_id) > 1})
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"{source} contains duplicate feature IDs: {preview}")
    return result


def canonical_ids_from_map_file(map_file: str | Path) -> list[str]:
    """Read PLINK MAP/BIM rows and return canonical IDs in file order."""
    feature_ids: list[str] = []
    with open(map_file, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 4:
                raise ValueError(f"Malformed MAP/BIM row {line_number}: expected 4 columns")
            feature_ids.append(canonical_snp_id_from_map(fields[0], fields[3]))
    return ensure_unique_feature_ids(feature_ids, source=str(map_file))
