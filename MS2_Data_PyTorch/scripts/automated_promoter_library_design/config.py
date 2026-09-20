# -*- coding: utf-8 -*-
"""The editable design configuration, its validation, and run-config output.

Entry point in 01_recursive_design.ipynb: Batch 0.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from ._common import ELEMENTS, ELEMENT_LENGTHS, SPACER_DERIVED_VERSION_IDX, _file_signature, _json_dump, _validated_fraction_range, normalize_dna


@dataclass(frozen=True)
class DesignUnitKey:
    element: str
    unit_id: str

    def label(self) -> str:
        return f"{self.element}:{self.unit_id}"


@dataclass
class DesignConfig:
    # Locked sequences per element, never touched by the search. Either one string
    # (the usual single consensus) or an ordered sequence of strings when an
    # element needs several locked slots. They occupy the versions above the last
    # mutable bin, weakest slot first: with N bins, entry i lands on v{N+1+i}, so
    # ["TGGACTGATATATACAAAA", "TAAAAAATTTGGAAAATAG"] with 3 bins gives locked v4
    # and v5. Normalised to a tuple per element by __post_init__.
    consensus: dict[str, str | Iterable[str]]
    bg5: str = "T" * 13
    bg3: str = "A" * 13
    gap_length: int = 3
    gap_seed: int = 777
    # Fallback bin count for elements absent from mutable_energy_fraction_ranges.
    n_energy_bins: int = 4
    default_energy_fraction_range: tuple[float, float] = (0.0, 0.8)
    max_candidates_per_unit: int = 60
    max_abs_shift: int = 2
    max_shift_rate: float = 0.10
    # Phase-switch trigger only, never an acceptance threshold: once the m10
    # shifted rate is at or below this, the search moves on to the -35 correction
    # phase instead of driving m10 all the way to max_shift_rate first. Both
    # anchors are still validated at max_shift_rate, so the phase returns to m10
    # after m35 passes to finish the job. Setting it equal to max_shift_rate gives
    # back the original single-pass m10 -> m35 schedule apart from two edges: the
    # gate is inclusive and ignores out-of-range variants, so a rate sitting
    # exactly on the threshold, or one below it with |shift| > 2 still present,
    # now enters the m35 phase where the old m10_validation_pass check did not.
    m10_phase_exit_rate: float = 0.20
    scan_batch_size: int = 1024
    require_derived_spacer_in_database: bool = False
    # Per-element pooling boundary: how much of the observed min-max energy axis
    # to bin, and into how many bins. Accepts (lower, upper) - which falls back to
    # n_energy_bins - or (lower, upper, n_bins). Every bin becomes one mutable
    # design version, so an element with N bins and L locked sequences contributes
    # versions v1..vN plus locked v{N+1}..v{N+L}, and the library holds
    # prod(N_e + L_e) variants.
    # Example: {"UP": (0.0, 0.8, 4), "m35": (0.3, 0.9, 4)}.
    mutable_energy_fraction_ranges: dict[
        str, tuple[float, float] | tuple[float, float, int]
    ] = field(default_factory=dict)
    # Drop candidates whose own measurements never cleared the assay's detection
    # limit. Below it a variant simply did not fluoresce, so its LogGFP carries no
    # information about how weak the element is - and those are exactly the
    # sequences that would otherwise define the weak end of the design ladder.
    #
    # detection_limit_gfp is on the LINEAR GFP scale that LogGFP is the log10 of;
    # None disables the filter and restores the pre-filter candidate pools.
    #
    # min_bright_measurements is clipped per sequence to how many rows it actually
    # has, which is what lets one rule cover two pool shapes: PL.pkl gives each
    # hexamer 19-25 rows (paired with many partners) so the clip does not bite and
    # the sequence must clear the limit in at least K of them, while UL/SL17/DL/ITS
    # give each sequence exactly one row, min(K, 1) = 1, and the rule degenerates to
    # "its own measurement was above the limit". Without the clip those four pools
    # would come back empty.
    detection_limit_gfp: float | None = 6.0
    min_bright_measurements: int = 5

    def __post_init__(self) -> None:
        missing = set(ELEMENTS) - set(self.consensus)
        if missing:
            raise ValueError(f"Missing consensus sequences: {sorted(missing)}")
        if self.n_energy_bins < 1:
            raise ValueError(f"n_energy_bins must be at least 1; received {self.n_energy_bins}")
        if self.detection_limit_gfp is not None:
            self.detection_limit_gfp = float(self.detection_limit_gfp)
            if not self.detection_limit_gfp > 0:
                raise ValueError(
                    "detection_limit_gfp is a linear GFP value and must be positive; "
                    f"received {self.detection_limit_gfp}. Use None to disable the filter."
                )
        if self.min_bright_measurements < 1:
            raise ValueError(
                "min_bright_measurements must be at least 1; received "
                f"{self.min_bright_measurements}"
            )
        if not 0.0 < self.m10_phase_exit_rate <= 1.0:
            raise ValueError(
                f"m10_phase_exit_rate must lie in (0, 1]; received {self.m10_phase_exit_rate}"
            )
        if self.m10_phase_exit_rate < self.max_shift_rate:
            raise ValueError(
                f"m10_phase_exit_rate ({self.m10_phase_exit_rate}) is stricter than "
                f"max_shift_rate ({self.max_shift_rate}), so m10 could pass validation "
                "while the phase gate still held the search in the m10 phase. Raise it "
                "to at least max_shift_rate."
            )
        self.default_energy_fraction_range = _validated_fraction_range(
            "default_energy_fraction_range", self.default_energy_fraction_range
        )
        normalized_ranges = {}
        for element, bounds in self.mutable_energy_fraction_ranges.items():
            if element not in ELEMENTS:
                raise ValueError(f"Unknown element in mutable energy range: {element!r}")
            if len(bounds) == 2:
                n_bins = self.n_energy_bins
            elif len(bounds) == 3:
                n_bins = int(bounds[2])
            else:
                raise ValueError(
                    f"Mutable energy range for {element} must be (lower, upper) or "
                    f"(lower, upper, n_bins), not {bounds!r}"
                )
            if n_bins < 1:
                raise ValueError(f"Mutable bin count for {element} must be at least 1; received {n_bins}")
            lower, upper = _validated_fraction_range(f"mutable energy range for {element}", bounds[:2])
            normalized_ranges[element] = (lower, upper, n_bins)
        self.mutable_energy_fraction_ranges = normalized_ranges
        spacer_bins = self.n_mutable_bins("spacer")
        if spacer_bins < SPACER_DERIVED_VERSION_IDX:
            raise ValueError(
                f"The spacer needs at least {SPACER_DERIVED_VERSION_IDX} bins because "
                f"v{SPACER_DERIVED_VERSION_IDX} is derived from v2 rather than sourced "
                f"from a bin; received {spacer_bins}"
            )
        normalized_consensus: dict[str, tuple[str, ...]] = {}
        for element, expected_len in ELEMENT_LENGTHS.items():
            entry = self.consensus[element]
            raw = (entry,) if isinstance(entry, str) else tuple(entry)
            if not raw:
                raise ValueError(f"Consensus {element} must hold at least one locked sequence")
            seqs = tuple(normalize_dna(seq) for seq in raw)
            for seq in seqs:
                if len(seq) != expected_len:
                    raise ValueError(
                        f"Consensus {element} has length {len(seq)}; expected {expected_len}: {seq}"
                    )
            if len(set(seqs)) != len(seqs):
                raise ValueError(f"Consensus {element} repeats a locked sequence: {seqs}")
            normalized_consensus[element] = seqs
        self.consensus = normalized_consensus
        self.bg5 = normalize_dna(self.bg5)
        self.bg3 = normalize_dna(self.bg3)

    def energy_bin_spec(self, element: str) -> tuple[int, float, float, str]:
        if element in self.mutable_energy_fraction_ranges:
            lower, upper, n_bins = self.mutable_energy_fraction_ranges[element]
            return n_bins, lower, upper, "custom_mutable_range"
        lower, upper = self.default_energy_fraction_range
        return self.n_energy_bins, lower, upper, "default_mutable_range"

    def n_mutable_bins(self, element: str) -> int:
        """Number of energy bins for this element; each bin is one mutable version."""
        if element in self.mutable_energy_fraction_ranges:
            return self.mutable_energy_fraction_ranges[element][2]
        return self.n_energy_bins

    def mutable_versions(self, element: str) -> tuple[str, ...]:
        return tuple(f"v{i}" for i in range(1, self.n_mutable_bins(element) + 1))

    def n_locked(self, element: str) -> int:
        """Number of locked sequences; each occupies one version above the bins."""
        return len(self.consensus[element])

    def locked_versions(self, element: str) -> tuple[str, ...]:
        """Locked slots, weakest first, filling the versions after the last bin."""
        first = self.n_mutable_bins(element) + 1
        return tuple(f"v{i}" for i in range(first, first + self.n_locked(element)))

    def locked_sequences(self, element: str) -> dict[str, str]:
        """Locked version -> sequence, in slot order."""
        return dict(zip(self.locked_versions(element), self.consensus[element]))

    def is_locked(self, element: str, version: str) -> bool:
        return version in self.locked_versions(element)

    def versions_for(self, element: str) -> tuple[str, ...]:
        return self.mutable_versions(element) + self.locked_versions(element)

    def n_variants(self) -> int:
        return math.prod(len(self.versions_for(element)) for element in ELEMENTS)


@dataclass
class DesignResult:
    success: bool
    stop_reason: str
    out_dir: Path
    final_state: dict[DesignUnitKey, int]
    final_elements: pd.DataFrame
    final_scan: pd.DataFrame
    final_summary: dict
    history: pd.DataFrame
    proposals: pd.DataFrame
    final_risk: pd.DataFrame
    final_overlap: pd.DataFrame


def save_run_config(
    out_dir: Path,
    config: DesignConfig,
    search_settings: dict,
    device: torch.device,
    core_model_checkpoint: Path | None = None,
    filename: str = "run_config.json",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "design_config": asdict(config),
        "search_settings": search_settings,
    }
    if core_model_checkpoint is not None:
        core_model_checkpoint = Path(core_model_checkpoint)
        if not core_model_checkpoint.exists():
            raise FileNotFoundError(f"Missing CorePromoter checkpoint: {core_model_checkpoint}")
        payload["core_model_checkpoint"] = _file_signature(core_model_checkpoint)
    _json_dump(out_dir / filename, payload)
