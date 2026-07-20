"""Feature-order manifests used to make trained models portable."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

FEATURE_MANIFEST_NAME = "feature_manifest.json"
FEATURE_ID_FORMAT = "chr<chrom>_<chromStart>_<chromEnd>"
_CANONICAL_ID_RE = re.compile(r"^chr[^_]+_\d+_\d+$")


def write_feature_manifest(
    directory: str | Path,
    feature_names: Iterable[object],
    *,
    source_file: str | None = None,
    filename: str = FEATURE_MANIFEST_NAME,
) -> Path:
    """Write an ordered feature manifest and return its path."""
    names = [str(name) for name in feature_names]
    if not names or len(names) != len(set(names)):
        raise ValueError("Feature manifest requires a non-empty list of unique feature names")
    path = Path(directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "feature_id_format": (
            FEATURE_ID_FORMAT
            if all(_CANONICAL_ID_RE.match(name) for name in names)
            else "custom"
        ),
        "feature_count": len(names),
        "feature_names": names,
    }
    if source_file:
        payload["source_file"] = str(source_file)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_feature_manifest(path: str | Path) -> list[str]:
    """Read and validate an ordered feature manifest."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = payload.get("feature_names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError(f"Invalid feature manifest: {path}")
    return [str(name) for name in names]


def find_feature_manifest(artifact_path: str | Path) -> Path | None:
    """Find a manifest next to a model artifact or in its result directory."""
    artifact = Path(artifact_path).resolve()
    candidates = []
    for parent in [artifact.parent, *artifact.parents]:
        candidates.append(parent / FEATURE_MANIFEST_NAME)
        candidates.append(parent / f"{artifact.stem}.features.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
