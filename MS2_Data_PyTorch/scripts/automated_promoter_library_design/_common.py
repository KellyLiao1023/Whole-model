# -*- coding: utf-8 -*-
"""Constants and small helpers shared by every other module here.

Entry point in 01_recursive_design.ipynb: used by every batch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import recursive_corepromoter_design as legacy


PROJECT_ROOT = legacy.PROJECT_ROOT


WEIGHTS_DIR = legacy.WEIGHTS_DIR


DEFAULT_PARENT_OUT = legacy.DEFAULT_PARENT_OUT


ELEMENTS = ("UP", "m35", "spacer", "m10", "DIS", "ITS")


# Spacer_v3 is derived as Spacer_v2[:-2] + "TG" instead of being sourced from its
# own energy bin, so bin 3 never contributes spacer candidates and v2/v3 form one
# coupled design unit. The spacer therefore needs at least this many bins.
SPACER_DERIVED_VERSION_IDX = 3


WINDOW_NAMES = (
    "W0_UP_1",
    "W1_UP_2",
    "W2_m35_spacer",
    "W3_spacer_mid",
    "W4_spacer_m10",
    "W5_m10_DIS",
    "W6_DIS_ITS",
    "W7_ITS_BG3",
)


ELEMENT_LENGTHS = {
    "UP": 19,
    "m35": 6,
    "spacer": 17,
    "m10": 6,
    "DIS": 8,
    "ITS": 10,
}


# Sequence lengths come from ELEMENT_LENGTHS; this table only says where to read.
SOURCE_SPECS = {
    "UP": ("UL.pkl", None),
    "m35": ("PL.pkl", "minus35"),
    "spacer": ("SL17.pkl", None),
    "m10": ("PL.pkl", "minus10"),
    "DIS": ("DL.pkl", None),
    "ITS": ("ITS.pkl", None),
}


# m35/m10 are scored by BPM, not by these two files. No code loads them: _load_all
# skips m35/m10, and build_scored_pools points those two at BPM/Params_Con17.pkl.
# They are kept, renamed _unused, because they rank hexamers differently from BPM
# (Spearman 0.37 / -0.22) and that disagreement is worth being able to reproduce.
WEIGHT_NAMES = {
    "UP": "weights_UP.pt",
    "m35": "weights_minus35_unused.pt",
    "m10": "weights_minus10_unused.pt",
    "DIS": "weights_Dis.pt",
    "ITS": "weights_ITS.pt",
}


def normalize_dna(seq: str) -> str:
    return "".join(str(seq).upper().split()).replace("U", "T")


def _validated_fraction_range(label: str, bounds) -> tuple[float, float]:
    lower, upper = (float(bounds[0]), float(bounds[1]))
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError(
            f"{label} must satisfy 0 <= lower < upper <= 1; received {(lower, upper)}"
        )
    return lower, upper


def version_index(version: str) -> int:
    """Sort key for version labels, so v10 orders after v9 rather than after v1."""
    return int(str(version)[1:])


def _bin_edges(
    e_min: float, e_max: float, lower_fraction: float, upper_fraction: float, n_bins: int
) -> np.ndarray:
    """Equal-width bin edges over the requested slice of an observed energy axis."""
    energy_span = e_max - e_min
    return np.linspace(
        e_min + lower_fraction * energy_span,
        e_min + upper_fraction * energy_span,
        n_bins + 1,
    )


def pool_bin_edges(pool: pd.DataFrame) -> tuple[np.ndarray, float, float, float, float, float, int]:
    """Rebuild a scored pool's exact bin edges from its constant columns.

    energy_bin_summary leaves energy_lower/energy_upper as NaN for empty bins, so
    anything that needs every edge recomputes them from energy_min/max and the
    stored fractions instead. Returns
    (edges, e_min, e_max, span, lower_fraction, upper_fraction, n_bins).
    """
    e_min = float(pool["energy_min"].iloc[0])
    e_max = float(pool["energy_max"].iloc[0])
    lower_fraction = float(pool["binning_fraction_lower"].iloc[0])
    upper_fraction = float(pool["binning_fraction_upper"].iloc[0])
    n_bins = int(pool["n_assigned_bins"].iloc[0])
    edges = _bin_edges(e_min, e_max, lower_fraction, upper_fraction, n_bins)
    return edges, e_min, e_max, e_max - e_min, lower_fraction, upper_fraction, n_bins


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_slice(seq: str, start: int, end: int) -> str:
    if start < 0 or end > len(seq) or start >= end:
        return ""
    return seq[start:end]


def _overlap_regions(intervals: list[tuple[str, int, int]], start: int, end: int) -> str:
    hits = [name for name, left, right in intervals if start < right and end > left]
    return "|".join(hits) if hits else "out"
