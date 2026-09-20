# -*- coding: utf-8 -*-
"""Element scoring, the sequence pools drawn from the PKL libraries, and energy binning.

Entry point in 01_recursive_design.ipynb: Batch 0.5 and Batch 1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

import recursive_corepromoter_design as legacy
from BPM.BPM import score_m10 as bpm_score_m10
from BPM.BPM import score_m35 as bpm_score_m35

from ._common import ELEMENTS, ELEMENT_LENGTHS, SOURCE_SPECS, WEIGHTS_DIR, WEIGHT_NAMES, _bin_edges, _file_signature, _json_dump, normalize_dna
from .config import DesignConfig


class ElementModelBundle:
    """Element models exposed on one higher-is-stronger score axis.

    The NN element models already use higher energy as stronger. Model_PL/BPM
    uses lower native dG as stronger, so m35/m10 return -dG here. This changes
    direction only; it does not calibrate scores across different elements.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.models: dict[str, legacy.ElementEnergyModel] = {}
        self.spacer_models: dict[int, legacy.ElementEnergyModel] = {}
        self._load_all()

    def _load_model(self, weight_path: Path, kernel_size: int) -> legacy.ElementEnergyModel:
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing trained element weight: {weight_path}")
        model = legacy.ElementEnergyModel(kernel_size).to(self.device)
        state = torch.load(weight_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.eval()

    def _load_all(self) -> None:
        for element, weight_name in WEIGHT_NAMES.items():
            if element in {"m35", "m10"}:
                continue
            self.models[element] = self._load_model(
                legacy.WEIGHTS_DIR / weight_name,
                ELEMENT_LENGTHS[element],
            )
        for length in (16, 17, 18):
            self.spacer_models[length] = self._load_model(
                legacy.WEIGHTS_DIR / f"weights_Sp{length}.pt",
                length,
            )

    @staticmethod
    def _corrected_energy(model: legacy.ElementEnergyModel, seqs: list[str], device: torch.device, batch_size: int) -> np.ndarray:
        if not seqs:
            return np.array([], dtype=float)
        correction = legacy.correction_term(model)
        values = []
        with torch.no_grad():
            for start in range(0, len(seqs), batch_size):
                batch = seqs[start : start + batch_size]
                x = torch.tensor(
                    np.array([legacy.dna_one_hot(seq) for seq in batch], dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )
                values.append(model.energy(x).detach().cpu().numpy() - correction)
        return np.concatenate(values).astype(float)

    def score(self, element: str, sequences: Iterable[str], batch_size: int = 4096) -> np.ndarray:
        seqs = [normalize_dna(seq) for seq in sequences]
        if element == "m35":
            return np.array([-float(bpm_score_m35(seq)) for seq in seqs], dtype=float)
        if element == "m10":
            return np.array([-float(bpm_score_m10(seq)) for seq in seqs], dtype=float)
        result = np.full(len(seqs), np.nan, dtype=float)
        if element == "spacer":
            for length, model in self.spacer_models.items():
                idx = [i for i, seq in enumerate(seqs) if len(seq) == length and set(seq) <= set("ACGT")]
                if idx:
                    vals = self._corrected_energy(model, [seqs[i] for i in idx], self.device, batch_size)
                    result[idx] = vals
            return result

        model = self.models[element]
        expected_len = ELEMENT_LENGTHS[element]
        idx = [i for i, seq in enumerate(seqs) if len(seq) == expected_len and set(seq) <= set("ACGT")]
        if idx:
            vals = self._corrected_energy(model, [seqs[i] for i in idx], self.device, batch_size)
            result[idx] = vals
        return result


def load_core_model(device: torch.device, checkpoint_path: Path | None = None) -> legacy.CorePromoterModel:
    checkpoint_path = checkpoint_path or legacy.WEIGHTS_DIR / "weights_CorePromoter_clean.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing CorePromoter checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = legacy.CorePromoterModel(
        seq_length=int(checkpoint["seq_length"]),
        num_conds=int(checkpoint["num_conds"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def _source_sequences(
    element: str,
    max_source_rows: int | None = None,
    detection_limit_gfp: float | None = None,
    min_bright_measurements: int = 1,
) -> tuple[list[str], Path]:
    filename, column = SOURCE_SPECS[element]
    expected_len = ELEMENT_LENGTHS[element]
    path = legacy.TABLE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_pickle(path)
    # Rebuilt on a fresh RangeIndex so the sequence column and LogGFP stay aligned
    # positionally: for the index-keyed pools the sequences come from df.index, which
    # would otherwise carry a different index than the LogGFP column.
    values = df.index if column is None else df[column]
    seqs = pd.Series(np.asarray(values), dtype="string").map(normalize_dna)
    usable = (
        seqs.notna()
        & (seqs.str.len() == expected_len)
        & seqs.str.fullmatch(r"[ACGT]+", na=False)
    )
    seqs = seqs[usable]

    if detection_limit_gfp is not None:
        if "LogGFP" not in df.columns:
            raise ValueError(
                f"{filename} has no LogGFP column, so the detection-limit filter "
                f"cannot be applied to {element}. Set detection_limit_gfp=None to "
                "disable it."
            )
        log_gfp = pd.to_numeric(pd.Series(np.asarray(df["LogGFP"])), errors="coerce")
        bright = (10.0 ** log_gfp)[usable.to_numpy()] >= detection_limit_gfp
        per_sequence = bright.groupby(seqs.to_numpy())
        # Clip to the rows a sequence actually has: single-measurement pools then
        # need only their own row to be bright, while PL's many-partner rows need K.
        required = per_sequence.size().clip(upper=min_bright_measurements)
        survivors = per_sequence.sum() >= required
        seqs = seqs[seqs.isin(survivors.index[survivors])]

    seqs = seqs.drop_duplicates()
    if max_source_rows is not None and len(seqs) > max_source_rows:
        seqs = seqs.sample(max_source_rows, random_state=777)
    return sorted(seqs.tolist()), path


def build_scored_pools(
    models: ElementModelBundle,
    config: DesignConfig,
    cache_dir: Path,
    use_cache: bool = True,
    max_source_rows: int | None = None,
) -> dict[str, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    pools: dict[str, pd.DataFrame] = {}
    for element in ELEMENTS:
        seqs, source_path = _source_sequences(
            element,
            max_source_rows=max_source_rows,
            detection_limit_gfp=config.detection_limit_gfp,
            min_bright_measurements=config.min_bright_measurements,
        )
        if element in {"m35", "m10"}:
            weight_path = legacy.SCRIPT_DIR / "BPM" / "Params_Con17.pkl"
        else:
            weight_name = f"weights_Sp{ELEMENT_LENGTHS['spacer']}.pt" if element == "spacer" else WEIGHT_NAMES[element]
            weight_path = legacy.WEIGHTS_DIR / weight_name
        # The detection-limit settings change which sequences are in the pool at all,
        # so they belong in the cache identity: without them a filtered run would
        # silently reuse an unfiltered pool built from the same source file.
        cache_path = cache_dir / f"scored_pool_{element}.pkl"
        meta_path = cache_dir / f"scored_pool_{element}.json"
        signature = {
            "element": element,
            "source": _file_signature(source_path),
            "weight": _file_signature(weight_path),
            "max_source_rows": max_source_rows,
            "detection_limit_gfp": config.detection_limit_gfp,
            "min_bright_measurements": config.min_bright_measurements,
        }
        cached_ok = False
        if use_cache and cache_path.exists() and meta_path.exists():
            try:
                cached_ok = json.loads(meta_path.read_text(encoding="utf-8")) == signature
            except json.JSONDecodeError:
                # A truncated meta file just means the cache is unusable; rebuild it.
                cached_ok = False
        if cached_ok:
            pool = pd.read_pickle(cache_path)
        else:
            energy = models.score(element, seqs)
            pool = pd.DataFrame({"element": element, "sequence": seqs, "energy": energy})
            pool = pool[np.isfinite(pool["energy"])].drop_duplicates("sequence").reset_index(drop=True)
            pool.to_pickle(cache_path)
            _json_dump(meta_path, signature)
        n_bins, lower_fraction, upper_fraction, binning_mode = config.energy_bin_spec(element)
        pools[element] = assign_equal_width_bins(
            pool,
            n_bins=n_bins,
            lower_fraction=lower_fraction,
            upper_fraction=upper_fraction,
            binning_mode=binning_mode,
        )
    return pools


def assign_equal_width_bins(
    pool: pd.DataFrame,
    n_bins: int,
    lower_fraction: float,
    upper_fraction: float,
    binning_mode: str,
) -> pd.DataFrame:
    # The fractions arrive already validated by DesignConfig.__post_init__.
    out = pool.copy()
    e_min = float(out["energy"].min())
    e_max = float(out["energy"].max())
    if not np.isfinite(e_min) or not np.isfinite(e_max) or e_min >= e_max:
        raise ValueError(f"Cannot create energy bins: E_min={e_min}, E_max={e_max}")
    energy_span = e_max - e_min
    edges = _bin_edges(e_min, e_max, lower_fraction, upper_fraction, n_bins)
    out["energy_bin"] = pd.cut(
        out["energy"],
        bins=edges,
        labels=list(range(1, n_bins + 1)),
        include_lowest=True,
    ).astype("Int64")
    out["energy_min"] = e_min
    out["energy_max"] = e_max
    out["energy_fraction"] = (out["energy"] - e_min) / energy_span
    out["binning_mode"] = binning_mode
    out["binning_fraction_lower"] = lower_fraction
    out["binning_fraction_upper"] = upper_fraction
    out["n_assigned_bins"] = n_bins
    out["bin_lower"] = out["energy_bin"].map({i: edges[i - 1] for i in range(1, n_bins + 1)})
    out["bin_upper"] = out["energy_bin"].map({i: edges[i] for i in range(1, n_bins + 1)})
    out["bin_center"] = (out["bin_lower"] + out["bin_upper"]) / 2.0
    out["distance_to_bin_center"] = (out["energy"] - out["bin_center"]).abs()
    return out.sort_values(["energy_bin", "distance_to_bin_center", "sequence"]).reset_index(drop=True)


def locked_energies(
    models: ElementModelBundle, config: DesignConfig
) -> dict[str, dict[str, float]]:
    """Per element, the energy of every locked sequence keyed by its version slot.

    The locked slots are v{n_bins+1} onwards, so they are only literally v5 for an
    element with four bins and one locked sequence.
    """
    out: dict[str, dict[str, float]] = {}
    for element in ELEMENTS:
        versions = config.locked_versions(element)
        scores = models.score(element, list(config.consensus[element]))
        out[element] = {v: float(s) for v, s in zip(versions, scores)}
    return out


def binding_locked_energy(locked: dict[str, dict[str, float]], element: str) -> tuple[str, float]:
    """The weakest locked slot: mutable candidates must stay below this one."""
    version = min(locked[element], key=lambda v: locked[element][v])
    return version, locked[element][version]


def energy_bin_summary(pools: dict[str, pd.DataFrame], config: DesignConfig, models: ElementModelBundle) -> pd.DataFrame:
    locked = locked_energies(models, config)
    rows = []
    for element, pool in pools.items():
        bind_version, bind_energy = binding_locked_energy(locked, element)
        n_bins, lower_fraction, upper_fraction, binning_mode = config.energy_bin_spec(element)
        fraction_edges = np.linspace(lower_fraction, upper_fraction, n_bins + 1)
        n_below_range = int((pool["energy_fraction"] < lower_fraction).sum())
        n_above_range = int((pool["energy_fraction"] > upper_fraction).sum())
        for bin_id in range(1, n_bins + 1):
            sub = pool[pool["energy_bin"] == bin_id]
            rows.append(
                {
                    "element": element,
                    "energy_bin": bin_id,
                    "binning_mode": binning_mode,
                    "energy_fraction_lower": float(fraction_edges[bin_id - 1]),
                    "energy_fraction_upper": float(fraction_edges[bin_id]),
                    "energy_lower": float(sub["bin_lower"].iloc[0]) if len(sub) else np.nan,
                    "energy_upper": float(sub["bin_upper"].iloc[0]) if len(sub) else np.nan,
                    "n_sequences": int(len(sub)),
                    "n_eligible_below_locked": int((sub["energy"] < bind_energy).sum()),
                    "n_excluded_below_range": n_below_range,
                    "n_excluded_above_range": n_above_range,
                    # The binding constraint is the weakest locked slot, which is
                    # not necessarily the top one once an element locks several.
                    "binding_locked_version": bind_version,
                    "binding_locked_energy": bind_energy,
                    "locked_energies": ";".join(
                        f"{v}={e:.4f}" for v, e in locked[element].items()
                    ),
                }
            )
    return pd.DataFrame(rows)
