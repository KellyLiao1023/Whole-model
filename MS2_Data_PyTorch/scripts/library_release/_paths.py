"""Shared paths, schema helpers and RE-site table for the library release pipeline.

Every notebook (01-07) starts with the same bootstrap cell, which locates this
file by walking up from the current working directory. After that all paths are
absolute, so it does not matter where Jupyter was launched from.

Import with::

    from _paths import *
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

__all__ = [
    "PROJECT_ROOT", "MS2_DIR", "SCRIPTS_DIR", "RELEASE_DIR",
    "DATA_DIR", "NATIVE_DIR", "PHAGE_DIR", "RELEASE_OUT",
    "TABLE_DIR", "WEIGHTS_DIR", "BPM_DIR", "DESIGN_OUT_ROOT",
    "STD_COLS", "RE_SITES", "revcomp", "has_re_site", "find_re_sites",
    "standardize", "latest_design_run", "require",
]


def _find_project_root(start: Path | None = None) -> Path:
    """Walk up until we see the MS2_Data_PyTorch/tables marker."""
    base = (Path.cwd() if start is None else Path(start)).resolve()
    for cand in [base, *base.parents]:
        if (cand / "MS2_Data_PyTorch" / "tables").is_dir():
            return cand
    raise FileNotFoundError(
        "Could not locate project root containing MS2_Data_PyTorch/tables "
        f"(searched upwards from {base})"
    )


PROJECT_ROOT = _find_project_root(Path(__file__).parent)
MS2_DIR = PROJECT_ROOT / "MS2_Data_PyTorch"
SCRIPTS_DIR = MS2_DIR / "scripts"
RELEASE_DIR = SCRIPTS_DIR / "library_release"

DATA_DIR = RELEASE_DIR / "data"
NATIVE_DIR = DATA_DIR / "native"
PHAGE_DIR = DATA_DIR / "phage"
# Named RELEASE_OUT, not OUT_DIR: notebook 01 already binds OUT_DIR to the
# design run directory, and a wildcard import must not shadow it.
RELEASE_OUT = RELEASE_DIR / "outputs"

TABLE_DIR = MS2_DIR / "tables"
WEIGHTS_DIR = MS2_DIR / "weights"
BPM_DIR = SCRIPTS_DIR / "BPM"
DESIGN_OUT_ROOT = PROJECT_ROOT / "outputs" / "design_runs"

RELEASE_OUT.mkdir(parents=True, exist_ok=True)

# The design libraries (automated_promoter_library_design, pas_library_qc, util,
# BPM) all live in scripts/ and are imported by name, so put that on the path
# once, here, instead of in every notebook.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# --------------------------------------------------------------------------
# Shared schema
# --------------------------------------------------------------------------
STD_COLS = [
    "candidate_id",
    "source",
    "promoter_sequence",
    "promoter_length",
    "alphabet_valid",
    "qc_pass",
]

_DNA = re.compile(r"^[ACGT]+$")


def require(path: Path, what: str = "input") -> Path:
    """Fail loudly and early rather than halfway through a notebook."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")
    return path


def standardize(
    df: pd.DataFrame,
    source: str,
    candidate_id: pd.Series | list | str,
    seq_col: str = "promoter_sequence",
    extra_pass: pd.Series | None = None,
) -> pd.DataFrame:
    """Attach the six shared columns to a per-source candidate table.

    ``candidate_id`` is either a ready-made series/list, or a prefix string in
    which case ids are ``{prefix}{row:05d}``. ``alphabet_valid`` is the only
    sequence-content gate applied here; ``extra_pass`` lets a source add its own
    condition (both must hold for ``qc_pass``).
    """
    out = df.copy().reset_index(drop=True)

    seq = out[seq_col].astype("string").str.strip().str.upper()
    out["promoter_sequence"] = seq
    out["promoter_length"] = seq.str.len().astype("Int64")
    out["alphabet_valid"] = seq.fillna("").str.match(_DNA).fillna(False)

    out["source"] = source
    if isinstance(candidate_id, str):
        out["candidate_id"] = [f"{candidate_id}{i:05d}" for i in range(1, len(out) + 1)]
    else:
        out["candidate_id"] = pd.Series(list(candidate_id), dtype="string").values

    ok = out["alphabet_valid"] & (out["promoter_length"] > 0)
    if extra_pass is not None:
        ok = ok & pd.Series(list(extra_pass), index=out.index).fillna(False)
    out["qc_pass"] = ok

    if out["candidate_id"].duplicated().any():
        dup = out.loc[out["candidate_id"].duplicated(), "candidate_id"].head().tolist()
        raise ValueError(f"Duplicate candidate_id generated for source={source}: {dup}")

    rest = [c for c in out.columns if c not in STD_COLS]
    return out[STD_COLS + rest]


def latest_design_run(filename: str = "final_scan_15625.csv") -> Path:
    """Newest automated_redesign_* directory that actually finished."""
    runs = sorted(
        (p for p in DESIGN_OUT_ROOT.glob("automated_redesign_*") if (p / filename).exists()),
        key=lambda p: p.name,
    )
    if not runs:
        raise FileNotFoundError(
            f"No completed design run containing {filename} under {DESIGN_OUT_ROOT}"
        )
    return runs[-1]


# --------------------------------------------------------------------------
# Restriction sites (single definition, used by 05 and 07)
# --------------------------------------------------------------------------
RE_SITES = {
    "BamHI": "GGATCC",
    "BcuI": "ACTAGT",     # = SpeI
    "BglII": "AGATCT",
    "Eco31I": "GGTCTC",   # = BsaI, non-palindromic -> revcomp GAGACC scanned too
    "EcoRI": "GAATTC",
    "HindIII": "AAGCTT",
    "KpnI": "GGTACC",
    "MluI": "ACGCGT",
    "NcoI": "CCATGG",
    "NdeI": "CATATG",
    "NheI": "GCTAGC",
    "NotI": "GCGGCCGC",
    "PstI": "CTGCAG",
    "SacI": "GAGCTC",
    "SalI": "GTCGAC",
    "SmaI": "CCCGGG",
    "VspI": "ATTAAT",     # = AseI
    "XbaI": "TCTAGA",
    "XhoI": "CTCGAG",
}

_COMP = str.maketrans("ACGT", "TGCA")


def revcomp(seq: str) -> str:
    return str(seq).upper().translate(_COMP)[::-1]


def find_re_sites(seq: str, site: str) -> list[int]:
    """0-based start positions of ``site`` and its reverse complement in ``seq``."""
    seq = str(seq).upper()
    positions: list[int] = []
    for pattern in {site, revcomp(site)}:
        start = 0
        while True:
            idx = seq.find(pattern, start)
            if idx < 0:
                break
            positions.append(idx)
            start = idx + 1
    return sorted(positions)


def has_re_site(seq: str, re_sites: dict[str, str] | None = None) -> bool:
    sites = RE_SITES if re_sites is None else re_sites
    return any(find_re_sites(seq, s) for s in sites.values())
