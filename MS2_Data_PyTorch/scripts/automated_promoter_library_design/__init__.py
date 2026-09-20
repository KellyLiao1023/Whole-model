# -*- coding: utf-8 -*-
"""Automated promoter library design.

This package is the index. Each module maps to one stage of
01_recursive_design.ipynb, so a batch you cannot follow has exactly one
file to open:

    _common.py    used by every batch      Constants and small helpers shared by every other module here.
    config.py     Batch 0                  The editable design configuration, its validation, and run-config output.
    scoring.py    Batch 0.5 and Batch 1    Element scoring, the sequence pools drawn from the PKL libraries, and energy binning.
    space.py      Batch 2 and Batch 3      The design space: units, candidate selection, gap assignment, library assembly.
    scanning.py   Batch 4 and Batch 5      Full-sequence CorePromoter scanning, validation summary, shift diagnosis.
    search.py     Batch 6                  The monotonic automated redesign search.

Notebooks import this package as `r` and reach everything through it,
so the names below are the supported surface.
"""
from __future__ import annotations

from ._common import (
    PROJECT_ROOT,
    WEIGHTS_DIR,
    DEFAULT_PARENT_OUT,
    ELEMENTS,
    SPACER_DERIVED_VERSION_IDX,
    WINDOW_NAMES,
    ELEMENT_LENGTHS,
    SOURCE_SPECS,
    WEIGHT_NAMES,
    normalize_dna,
    _validated_fraction_range,
    version_index,
    _bin_edges,
    pool_bin_edges,
    _file_signature,
    _json_dump,
    _safe_slice,
    _overlap_regions,
)
from .config import (
    DesignUnitKey,
    DesignConfig,
    DesignResult,
    save_run_config,
)
from .scoring import (
    ElementModelBundle,
    load_core_model,
    _source_sequences,
    build_scored_pools,
    assign_equal_width_bins,
    locked_energies,
    binding_locked_energy,
    energy_bin_summary,
)
from .space import (
    DesignSpace,
    build_balanced_gap_assignment,
    assemble_library,
)
from .scanning import (
    CorePromoterScanner,
    summarize_validation,
    objective_for_phase,
    _dominant_shift,
    diagnose_shift_drivers,
)
from .search import (
    AutomatedRedesigner,
)

__all__ = [
    "PROJECT_ROOT",
    "WEIGHTS_DIR",
    "DEFAULT_PARENT_OUT",
    "ELEMENTS",
    "SPACER_DERIVED_VERSION_IDX",
    "WINDOW_NAMES",
    "ELEMENT_LENGTHS",
    "SOURCE_SPECS",
    "WEIGHT_NAMES",
    "normalize_dna",
    "_validated_fraction_range",
    "version_index",
    "_bin_edges",
    "pool_bin_edges",
    "_file_signature",
    "_json_dump",
    "_safe_slice",
    "_overlap_regions",
    "DesignUnitKey",
    "DesignConfig",
    "DesignResult",
    "save_run_config",
    "ElementModelBundle",
    "load_core_model",
    "_source_sequences",
    "build_scored_pools",
    "assign_equal_width_bins",
    "locked_energies",
    "binding_locked_energy",
    "energy_bin_summary",
    "DesignSpace",
    "build_balanced_gap_assignment",
    "assemble_library",
    "CorePromoterScanner",
    "summarize_validation",
    "objective_for_phase",
    "_dominant_shift",
    "diagnose_shift_drivers",
    "AutomatedRedesigner",
]
