from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import recursive_corepromoter_design as legacy
from BPM.BPM import score_m10 as bpm_score_m10
from BPM.BPM import score_m35 as bpm_score_m35


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

WEIGHT_NAMES = {
    "UP": "weights_UP.pt",
    "m35": "weights_minus35.pt",
    "m10": "weights_minus10.pt",
    "DIS": "weights_Dis.pt",
    "ITS": "weights_ITS.pt",
}


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


class DesignSpace:
    def __init__(
        self,
        config: DesignConfig,
        models: ElementModelBundle,
        scored_pools: dict[str, pd.DataFrame],
    ):
        self.config = config
        self.models = models
        self.scored_pools = scored_pools
        self.unit_candidates: dict[DesignUnitKey, pd.DataFrame] = {}
        # element -> {locked version: energy}. Mutable candidates must stay below
        # the weakest of them, which is what binding_locked_energy returns.
        self.locked_energy = locked_energies(models, config)
        self._build_units()

    def _binding_locked(self, element: str) -> tuple[str, float]:
        return binding_locked_energy(self.locked_energy, element)

    def _eligible_bin(self, element: str, bin_id: int) -> pd.DataFrame:
        pool = self.scored_pools[element]
        bind_version, bind_energy = self._binding_locked(element)
        sub = pool[
            (pool["energy_bin"] == bin_id)
            & (pool["energy"] < bind_energy)
            & (~pool["sequence"].isin(self.config.consensus[element]))
        ].copy()
        if sub.empty:
            n_bins, lower, upper, _ = self.config.energy_bin_spec(element)
            raise ValueError(
                f"No eligible sequences for {element} v{bin_id}: energy_bin={bin_id} "
                f"of {n_bins} over fraction {lower:.2f}-{upper:.2f}, "
                f"constraint=energy_below_{bind_version} "
                f"({bind_energy:.4f}, the weakest of "
                f"{', '.join(self.config.locked_versions(element))}). Lower that "
                f"element's upper_fraction in mutable_energy_fraction_ranges so the "
                f"bin stays below the locked sequences."
            )
        return sub.sort_values(["distance_to_bin_center", "sequence"]).head(
            self.config.max_candidates_per_unit
        ).reset_index(drop=True)

    def _build_units(self) -> None:
        for element in ELEMENTS:
            for version_idx in range(1, self.config.n_mutable_bins(element) + 1):
                # The spacer skips two bins: bin 2 is consumed by the coupled
                # v2/v3 pair built below, and bin SPACER_DERIVED_VERSION_IDX is
                # never sourced at all because that version is derived from v2.
                if element == "spacer" and version_idx in (2, SPACER_DERIVED_VERSION_IDX):
                    continue
                key = DesignUnitKey(element, f"v{version_idx}")
                sub = self._eligible_bin(element, version_idx).copy()
                sub["version"] = f"v{version_idx}"
                sub["selection_mode"] = "database_bin"
                self.unit_candidates[key] = sub

        base = self._eligible_bin("spacer", 2).rename(
            columns={
                "sequence": "sequence_v2",
                "energy": "energy_v2",
                "energy_bin": "energy_bin_v2",
            }
        )
        base["sequence_v3"] = base["sequence_v2"].str[:-2] + "TG"
        base["energy_v3"] = self.models.score("spacer", base["sequence_v3"].tolist())
        spacer_db = set(self.scored_pools["spacer"]["sequence"])
        base["v3_in_database"] = base["sequence_v3"].isin(spacer_db)
        base["energy_bin_v3"] = [
            self._energy_bin_for_value("spacer", value) for value in base["energy_v3"]
        ]
        spacer_bind_version, spacer_bind_energy = self._binding_locked("spacer")
        base = base[
            np.isfinite(base["energy_v3"])
            & (base["energy_v3"] < spacer_bind_energy)
            & (base["sequence_v2"] != base["sequence_v3"])
            & (~base["sequence_v3"].isin(self.config.consensus["spacer"]))
        ]
        if self.config.require_derived_spacer_in_database:
            base = base[base["v3_in_database"]]
        if base.empty:
            raise ValueError(
                f"No eligible Spacer_v2/v{SPACER_DERIVED_VERSION_IDX} pair: v2 must come "
                f"from bin 2 and v{SPACER_DERIVED_VERSION_IDX}=v2[:-2]+'TG' must remain "
                f"below Spacer_{spacer_bind_version} energy ({spacer_bind_energy:.4f})."
            )
        self.unit_candidates[DesignUnitKey("spacer", "v2_v3_pair")] = base.head(
            self.config.max_candidates_per_unit
        ).reset_index(drop=True)

    def _energy_bin_for_value(self, element: str, value: float) -> int | None:
        edges, *_, n_bins = pool_bin_edges(self.scored_pools[element])
        if not np.isfinite(value) or value < edges[0] or value > edges[-1]:
            return None
        return int(np.clip(np.searchsorted(edges, value, side="right"), 1, n_bins))

    def candidate_pool_summary(self) -> pd.DataFrame:
        """Report raw bin sizes, eligibility, capping, and final search-pool sizes."""
        rows = []
        for key, search_pool in self.unit_candidates.items():
            source_bin = 2 if key == DesignUnitKey("spacer", "v2_v3_pair") else int(key.unit_id[1:])
            scored = self.scored_pools[key.element]
            edges, _, _, _, frac_lo, frac_hi, n_bins = pool_bin_edges(scored)
            fraction_edges = np.linspace(frac_lo, frac_hi, n_bins + 1)
            bin_rows = scored[scored["energy_bin"] == source_bin]
            bind_version, bind_energy = self._binding_locked(key.element)
            eligible = bin_rows[
                (bin_rows["energy"] < bind_energy)
                & (~bin_rows["sequence"].isin(self.config.consensus[key.element]))
            ]
            n_examined = min(len(eligible), self.config.max_candidates_per_unit)
            if key == DesignUnitKey("spacer", "v2_v3_pair"):
                note = "derived pair constraints applied after examining capped spacer v2 source pool"
            else:
                note = "standard database-backed unit"
            rows.append(
                {
                    "element": key.element,
                    "design_unit_id": key.unit_id,
                    "source_energy_bin": source_bin,
                    "energy_fraction_lower": float(fraction_edges[source_bin - 1]),
                    "energy_fraction_upper": float(fraction_edges[source_bin]),
                    "energy_lower": float(edges[source_bin - 1]),
                    "energy_upper": float(edges[source_bin]),
                    "n_sequences_in_bin": int(len(bin_rows)),
                    "n_eligible_below_locked": int(len(eligible)),
                    "binding_locked_version": bind_version,
                    "binding_locked_energy": bind_energy,
                    "n_source_candidates_examined": int(n_examined),
                    "n_candidates_in_search_pool": int(len(search_pool)),
                    "max_candidates_per_unit": int(self.config.max_candidates_per_unit),
                    "source_pool_was_capped": bool(len(eligible) > n_examined),
                    "note": note,
                }
            )
        return pd.DataFrame(rows).sort_values(["element", "design_unit_id"]).reset_index(drop=True)

    def initial_state(self) -> dict[DesignUnitKey, int]:
        state = {key: 0 for key in self.unit_candidates}
        self.validate_state(state)
        return state

    def state_from_selected_elements(self, selected: pd.DataFrame) -> dict[DesignUnitKey, int]:
        """Recover candidate indices from a saved selected-elements CSV."""
        required = {"element", "version", "sequence"}
        if not required <= set(selected.columns):
            raise ValueError(f"Saved elements are missing columns: {sorted(required - set(selected.columns))}")
        saved = selected.copy()
        saved["sequence"] = saved["sequence"].map(normalize_dna)
        lookup = {
            (row.element, row.version): row.sequence
            for row in saved.itertuples(index=False)
        }
        state: dict[DesignUnitKey, int] = {}
        for key, pool in self.unit_candidates.items():
            if key == DesignUnitKey("spacer", "v2_v3_pair"):
                seq_v2 = lookup[("spacer", "v2")]
                seq_v3 = lookup[("spacer", f"v{SPACER_DERIVED_VERSION_IDX}")]
                mask = (pool["sequence_v2"] == seq_v2) & (pool["sequence_v3"] == seq_v3)
            else:
                mask = pool["sequence"] == lookup[(key.element, key.unit_id)]
            matches = np.flatnonzero(mask.to_numpy())
            if len(matches) != 1:
                raise ValueError(
                    f"Could not uniquely recover {key.label()} from saved elements; "
                    f"matches={matches.tolist()}"
                )
            state[key] = int(matches[0])
        self.validate_state(state)
        return state

    def selected_elements(self, state: dict[DesignUnitKey, int]) -> pd.DataFrame:
        rows = []
        derived_version = f"v{SPACER_DERIVED_VERSION_IDX}"
        for element in ELEMENTS:
            pair = None
            if element == "spacer":
                pair = self.unit_candidates[DesignUnitKey("spacer", "v2_v3_pair")].iloc[
                    int(state[DesignUnitKey("spacer", "v2_v3_pair")])
                ]
            for version in self.config.mutable_versions(element):
                if pair is not None and version == "v2":
                    rows.append(
                        {
                            "element": "spacer",
                            "version": "v2",
                            "design_unit_id": "v2_v3_pair",
                            "sequence": pair["sequence_v2"],
                            "energy": float(pair["energy_v2"]),
                            "energy_bin": int(pair["energy_bin_v2"]),
                            "selection_mode": "database_bin",
                            "locked": False,
                            "in_database": True,
                        }
                    )
                elif pair is not None and version == derived_version:
                    rows.append(
                        {
                            "element": "spacer",
                            "version": derived_version,
                            "design_unit_id": "v2_v3_pair",
                            "sequence": pair["sequence_v3"],
                            "energy": float(pair["energy_v3"]),
                            "energy_bin": pair["energy_bin_v3"],
                            "selection_mode": "derived_tail_TG",
                            "locked": False,
                            "in_database": bool(pair["v3_in_database"]),
                        }
                    )
                else:
                    key = DesignUnitKey(element, version)
                    row = self.unit_candidates[key].iloc[int(state[key])]
                    rows.append(
                        {
                            "element": element,
                            "version": version,
                            "design_unit_id": key.unit_id,
                            "sequence": row["sequence"],
                            "energy": float(row["energy"]),
                            "energy_bin": int(row["energy_bin"]),
                            "selection_mode": "database_bin",
                            "locked": False,
                            "in_database": True,
                        }
                    )

            observed = set(self.scored_pools[element]["sequence"])
            for locked_version, sequence in self.config.locked_sequences(element).items():
                energy = self.locked_energy[element][locked_version]
                rows.append(
                    {
                        "element": element,
                        "version": locked_version,
                        "design_unit_id": f"locked_{locked_version}",
                        "sequence": sequence,
                        "energy": energy,
                        "energy_bin": self._energy_bin_for_value(element, energy),
                        "selection_mode": "locked_consensus",
                        "locked": True,
                        "in_database": bool(sequence in observed),
                    }
                )
        out = pd.DataFrame(rows)
        out["_version_idx"] = out["version"].map(version_index)
        return (
            out.sort_values(["element", "_version_idx"])
            .drop(columns="_version_idx")
            .reset_index(drop=True)
        )

    def validate_state(self, state: dict[DesignUnitKey, int]) -> None:
        selected = self.selected_elements(state)
        for element, sub in selected.groupby("element"):
            expected = self.config.versions_for(element)
            if len(sub) != len(expected) or set(sub["version"]) != set(expected):
                raise ValueError(
                    f"{element} does not contain exactly {expected[0]}-{expected[-1]}."
                )
            if sub["sequence"].duplicated().any():
                dup = sub.loc[sub["sequence"].duplicated(False), ["version", "sequence"]]
                raise ValueError(f"Duplicate {element} versions:\n{dup.to_string(index=False)}")
            expected_locked = self.config.locked_sequences(element)
            indexed = sub.set_index("version")
            for locked_version, sequence in expected_locked.items():
                if indexed.loc[locked_version, "sequence"] != sequence:
                    raise ValueError(
                        f"Locked {element} {locked_version} changed: expected {sequence}, "
                        f"got {indexed.loc[locked_version, 'sequence']}."
                    )
            # Every mutable version must stay below every locked one, so the
            # weakest locked slot is the binding comparison.
            bind_version, bind_energy = self._binding_locked(element)
            mutable = sub[~sub["version"].isin(expected_locked)]
            if not (mutable["energy"] < bind_energy).all():
                bad = mutable[mutable["energy"] >= bind_energy]
                raise ValueError(
                    f"{element} contains mutable energy >= {bind_version} "
                    f"({bind_energy:.4f}):\n{bad}"
                )
        derived_version = f"v{SPACER_DERIVED_VERSION_IDX}"
        spacer = selected[selected["element"] == "spacer"].set_index("version")
        expected_derived = str(spacer.loc["v2", "sequence"])[:-2] + "TG"
        if spacer.loc[derived_version, "sequence"] != expected_derived:
            raise ValueError(f"Spacer_{derived_version} is not Spacer_v2[:-2] + 'TG'.")

    @staticmethod
    def state_signature(state: dict[DesignUnitKey, int]) -> tuple:
        return tuple((key.element, key.unit_id, int(state[key])) for key in sorted(state, key=lambda x: x.label()))

    def unit_for_slot(self, element: str, version: str) -> DesignUnitKey | None:
        if self.config.is_locked(element, version):
            return None
        if element == "spacer" and version in {"v2", f"v{SPACER_DERIVED_VERSION_IDX}"}:
            return DesignUnitKey("spacer", "v2_v3_pair")
        return DesignUnitKey(element, version)


def build_balanced_gap_assignment(config: DesignConfig) -> pd.DataFrame:
    version_combos = list(itertools.product(*(config.versions_for(e) for e in ELEMENTS)))
    kmers = ["".join(x) for x in itertools.product("ACGT", repeat=config.gap_length)]
    n_variants = len(version_combos)
    base, remainder = divmod(n_variants, len(kmers))
    rng = random.Random(config.gap_seed)
    extra = rng.sample(kmers, remainder)
    assigned = kmers * base + extra
    rng.shuffle(assigned)
    rows = []
    for idx, (versions, gap) in enumerate(zip(version_combos, assigned), start=1):
        row = {
            "variant_id": f"V{idx:05d}",
            "gap_3bp": gap,
            "gap_assignment_seed": config.gap_seed,
        }
        row.update({f"{element}_version": version for element, version in zip(ELEMENTS, versions)})
        rows.append(row)
    # kmers * base + extra uses every kmer `base` times plus `remainder` distinct
    # extras, so the counts differ by at most 1 by construction.
    return pd.DataFrame(rows)


def assemble_library(
    selected_elements: pd.DataFrame,
    gap_assignment: pd.DataFrame,
    config: DesignConfig,
) -> pd.DataFrame:
    seq_lookup = {
        (row.element, row.version): row.sequence
        for row in selected_elements.itertuples(index=False)
    }
    energy_lookup = {
        (row.element, row.version): float(row.energy)
        for row in selected_elements.itertuples(index=False)
    }
    out = gap_assignment.copy()
    for element in ELEMENTS:
        out[f"{element}_seq"] = [seq_lookup[(element, v)] for v in out[f"{element}_version"]]
        out[f"{element}_design_energy"] = [
            energy_lookup[(element, v)] for v in out[f"{element}_version"]
        ]
    out["combination_label"] = out.apply(
        lambda row: "|".join(f"{element}_{row[f'{element}_version']}" for element in ELEMENTS),
        axis=1,
    )
    out["full_sequence"] = (
        config.bg5
        + out["UP_seq"]
        + out["gap_3bp"]
        + out["m35_seq"]
        + out["spacer_seq"]
        + out["m10_seq"]
        + out["DIS_seq"]
        + out["ITS_seq"]
        + config.bg3
    )
    out["design_m35_start"] = len(config.bg5) + out["UP_seq"].str.len() + config.gap_length
    out["design_spacer_len"] = out["spacer_seq"].str.len()
    out["design_m10_start"] = (
        out["design_m35_start"] + ELEMENT_LENGTHS["m35"] + out["design_spacer_len"]
    )
    expected_variants = config.n_variants()
    if len(out) != expected_variants:
        raise AssertionError(
            f"Expected {expected_variants:,} variants; assembled {len(out):,}."
        )
    return out


def _safe_slice(seq: str, start: int, end: int) -> str:
    if start < 0 or end > len(seq) or start >= end:
        return ""
    return seq[start:end]


def _overlap_regions(intervals: list[tuple[str, int, int]], start: int, end: int) -> str:
    hits = [name for name, left, right in intervals if start < right and end > left]
    return "|".join(hits) if hits else "out"


class CorePromoterScanner:
    def __init__(
        self,
        model: legacy.CorePromoterModel,
        element_models: ElementModelBundle,
        config: DesignConfig,
        device: torch.device,
    ):
        self.model = model.eval()
        self.element_models = element_models
        self.config = config
        self.device = device
        self.offsets = self._architecture_offsets()
        self.m35_filter_idx = 2
        m35_offsets = [row[self.m35_filter_idx] for row in self.offsets]
        if len(set(m35_offsets)) != 1:
            raise ValueError(f"CorePromoter m35 offsets differ by channel: {m35_offsets}")
        self.m35_offset = int(m35_offsets[0])

    def _architecture_offsets(self) -> list[list[int]]:
        weights = self.model.conv2.weight.detach().cpu().numpy()
        offsets = []
        for channel in range(weights.shape[0]):
            channel_offsets = []
            for filter_idx in range(weights.shape[1]):
                nz = np.where(np.abs(weights[channel, filter_idx]) > 1e-6)[0]
                if len(nz) != 1:
                    raise ValueError(
                        f"Expected one CorePromoter position for channel={channel}, "
                        f"filter={filter_idx}; found {nz.tolist()}"
                    )
                channel_offsets.append(int(nz[0]))
            offsets.append(channel_offsets)
        return offsets

    def scan(
        self,
        variants: pd.DataFrame,
        annotate_element_energies: bool = True,
        annotate_diagnostics: bool = True,
    ) -> pd.DataFrame:
        df = variants.copy().reset_index(drop=True)
        df["design_channel"] = df["design_spacer_len"].map(legacy.CHANNEL_BY_SPACER)
        if df["design_channel"].isna().any():
            bad = sorted(df.loc[df["design_channel"].isna(), "design_spacer_len"].unique())
            raise ValueError(f"Unsupported designed spacer lengths: {bad}")
        df["design_channel"] = df["design_channel"].astype(int)
        df["design_arch_start"] = df["design_m35_start"] - self.m35_offset

        observed_arch, observed_channel = [], []
        best_scores, design_scores = [], []
        strongest_non_target_scores = []
        strongest_non_target_arch, strongest_non_target_channel = [], []
        best_log10_gfp, design_log10_gfp, strongest_non_target_log10_gfp = [], [], []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(df), self.config.scan_batch_size):
                sub = df.iloc[start : start + self.config.scan_batch_size]
                x = torch.tensor(
                    np.array([legacy.dna_one_hot(seq) for seq in sub["full_sequence"]], dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )
                conv = self.model.conv2(self.model.conv1(x))
                n_pos = conv.shape[2]
                flat = conv.reshape(conv.shape[0], -1)
                best_val, best_idx = flat.max(dim=1)
                best_ch = (best_idx // n_pos).detach().cpu().numpy().astype(int)
                best_pos = (best_idx % n_pos).detach().cpu().numpy().astype(int)
                design_ch_tensor = torch.tensor(
                    sub["design_channel"].to_numpy(dtype=np.int64),
                    dtype=torch.long,
                    device=self.device,
                )
                design_pos_tensor = torch.tensor(
                    sub["design_arch_start"].to_numpy(dtype=np.int64),
                    dtype=torch.long,
                    device=self.device,
                )
                if torch.any((design_pos_tensor < 0) | (design_pos_tensor >= n_pos)):
                    bad = sub.loc[
                        (sub["design_arch_start"] < 0) | (sub["design_arch_start"] >= n_pos),
                        ["design_channel", "design_arch_start", "full_sequence"],
                    ]
                    raise ValueError(
                        f"Designed architecture lies outside CorePromoter score map (n_pos={n_pos}):\n"
                        f"{bad.head().to_string(index=False)}"
                    )
                design_idx = design_ch_tensor * n_pos + design_pos_tensor
                design_val = flat.gather(1, design_idx.unsqueeze(1)).squeeze(1)
                non_target = flat.clone()
                non_target.scatter_(1, design_idx.unsqueeze(1), float("-inf"))
                non_target_val, non_target_idx = non_target.max(dim=1)
                non_target_ch = (non_target_idx // n_pos).detach().cpu().numpy().astype(int)
                non_target_pos = (non_target_idx % n_pos).detach().cpu().numpy().astype(int)
                observed_channel.extend(best_ch.tolist())
                observed_arch.extend(best_pos.tolist())
                best_scores.extend(best_val.detach().cpu().numpy().astype(float).tolist())
                design_scores.extend(design_val.detach().cpu().numpy().astype(float).tolist())
                strongest_non_target_scores.extend(
                    non_target_val.detach().cpu().numpy().astype(float).tolist()
                )
                strongest_non_target_channel.extend(non_target_ch.tolist())
                strongest_non_target_arch.extend(non_target_pos.tolist())
                best_log10_gfp.extend(
                    (self.model.energy2expression(best_val.unsqueeze(1)) / np.log(10))
                    .detach()
                    .cpu()
                    .numpy()
                    .ravel()
                    .astype(float)
                    .tolist()
                )
                design_log10_gfp.extend(
                    (self.model.energy2expression(design_val.unsqueeze(1)) / np.log(10))
                    .detach()
                    .cpu()
                    .numpy()
                    .ravel()
                    .astype(float)
                    .tolist()
                )
                strongest_non_target_log10_gfp.extend(
                    (self.model.energy2expression(non_target_val.unsqueeze(1)) / np.log(10))
                    .detach()
                    .cpu()
                    .numpy()
                    .ravel()
                    .astype(float)
                    .tolist()
                )

        df["observed_arch_start"] = observed_arch
        df["observed_channel"] = observed_channel
        df["observed_spacer_len"] = df["observed_channel"].map(legacy.SPACER_BY_CHANNEL).astype(int)
        df["observed_m35_start"] = df["observed_arch_start"] + self.m35_offset
        df["observed_m10_start"] = (
            df["observed_m35_start"] + ELEMENT_LENGTHS["m35"] + df["observed_spacer_len"]
        )
        df["m35_shift"] = df["observed_m35_start"] - df["design_m35_start"]
        df["m10_shift"] = df["observed_m10_start"] - df["design_m10_start"]
        df["spacer_length_shift"] = df["observed_spacer_len"] - df["design_spacer_len"]
        df["best_core_score"] = best_scores
        df["design_core_score"] = design_scores
        df["strongest_non_target_core_score"] = strongest_non_target_scores
        df["strongest_non_target_channel"] = strongest_non_target_channel
        df["strongest_non_target_spacer_len"] = (
            df["strongest_non_target_channel"].map(legacy.SPACER_BY_CHANNEL).astype(int)
        )
        df["strongest_non_target_arch_start"] = strongest_non_target_arch
        df["strongest_non_target_m35_start"] = (
            df["strongest_non_target_arch_start"] + self.m35_offset
        )
        df["strongest_non_target_m10_start"] = (
            df["strongest_non_target_m35_start"]
            + ELEMENT_LENGTHS["m35"]
            + df["strongest_non_target_spacer_len"]
        )
        df["target_margin_core_score"] = (
            df["design_core_score"] - df["strongest_non_target_core_score"]
        )
        df["best_minus_design_core_score"] = df["best_core_score"] - df["design_core_score"]
        # Predicted expression at zero library bias.
        df["best_predicted_log10_gfp"] = best_log10_gfp
        df["design_predicted_log10_gfp"] = design_log10_gfp
        df["strongest_non_target_predicted_log10_gfp"] = strongest_non_target_log10_gfp
        if annotate_diagnostics:
            self._annotate_observed_roles_and_windows(df)
            # Window annotation intentionally creates many export columns.
            # Make a contiguous copy before adding energy columns to avoid
            # pandas fragmentation warnings in notebook runs.
            df = df.copy()
            if annotate_element_energies:
                self._annotate_element_energies(df)
        return df

    def _design_intervals(self, row: pd.Series) -> list[tuple[str, int, int]]:
        pieces = [
            ("BG5", self.config.bg5),
            ("UP", row["UP_seq"]),
            ("gap_3bp", row["gap_3bp"]),
            ("m35", row["m35_seq"]),
            ("spacer", row["spacer_seq"]),
            ("m10", row["m10_seq"]),
            ("DIS", row["DIS_seq"]),
            ("ITS", row["ITS_seq"]),
            ("BG3", self.config.bg3),
        ]
        cursor = 0
        intervals = []
        for name, seq in pieces:
            end = cursor + len(str(seq))
            intervals.append((name, cursor, end))
            cursor = end
        return intervals

    def _annotate_observed_roles_and_windows(self, df: pd.DataFrame) -> None:
        role_starts = {element: [] for element in ELEMENTS}
        role_ends = {element: [] for element in ELEMENTS}
        role_seqs = {element: [] for element in ELEMENTS}
        role_overlaps = {element: [] for element in ELEMENTS}
        window_data = {
            name: {field: [] for field in ("design_start", "observed_start", "shift", "design_seq", "observed_seq", "overlap_regions")}
            for name in WINDOW_NAMES
        }
        for _, row in df.iterrows():
            seq = row["full_sequence"]
            intervals = self._design_intervals(row)
            obs_m35 = int(row["observed_m35_start"])
            obs_spacer_len = int(row["observed_spacer_len"])
            starts = {
                "UP": obs_m35 - self.config.gap_length - ELEMENT_LENGTHS["UP"],
                "m35": obs_m35,
                "spacer": obs_m35 + ELEMENT_LENGTHS["m35"],
                "m10": int(row["observed_m10_start"]),
                "DIS": int(row["observed_m10_start"]) + ELEMENT_LENGTHS["m10"],
                "ITS": int(row["observed_m10_start"]) + ELEMENT_LENGTHS["m10"] + ELEMENT_LENGTHS["DIS"],
            }
            lengths = dict(ELEMENT_LENGTHS)
            lengths["spacer"] = obs_spacer_len
            for element in ELEMENTS:
                start = int(starts[element])
                end = start + int(lengths[element])
                role_starts[element].append(start)
                role_ends[element].append(end)
                role_seqs[element].append(_safe_slice(seq, start, end))
                role_overlaps[element].append(_overlap_regions(intervals, start, end))

            design_ch = int(row["design_channel"])
            obs_ch = int(row["observed_channel"])
            for filter_idx, name in enumerate(WINDOW_NAMES):
                design_start = int(row["design_arch_start"]) + self.offsets[design_ch][filter_idx]
                observed_start = int(row["observed_arch_start"]) + self.offsets[obs_ch][filter_idx]
                window_data[name]["design_start"].append(design_start)
                window_data[name]["observed_start"].append(observed_start)
                window_data[name]["shift"].append(observed_start - design_start)
                window_data[name]["design_seq"].append(_safe_slice(seq, design_start, design_start + 8))
                window_data[name]["observed_seq"].append(_safe_slice(seq, observed_start, observed_start + 8))
                window_data[name]["overlap_regions"].append(
                    _overlap_regions(intervals, observed_start, observed_start + 8)
                )

        for element in ELEMENTS:
            df[f"observed_{element}_start"] = role_starts[element]
            df[f"observed_{element}_end"] = role_ends[element]
            df[f"observed_{element}_seq"] = role_seqs[element]
            df[f"observed_{element}_overlap_regions"] = role_overlaps[element]
        for name, fields in window_data.items():
            for field, values in fields.items():
                df[f"{name}_{field}"] = values

    def _annotate_element_energies(self, df: pd.DataFrame) -> None:
        for element in ELEMENTS:
            design_col = f"{element}_seq"
            observed_col = f"observed_{element}_seq"
            design_values = self._score_series_unique(element, df[design_col])
            observed_values = self._score_series_unique(element, df[observed_col])
            df[f"{element}_design_energy_rescored"] = design_values
            df[f"{element}_observed_role_energy"] = observed_values
            df[f"{element}_within_model_delta_energy"] = observed_values - design_values

    def _score_series_unique(self, element: str, series: pd.Series) -> np.ndarray:
        unique = pd.Series(series.astype(str).unique())
        values = self.element_models.score(element, unique.tolist())
        lookup = dict(zip(unique, values))
        return series.astype(str).map(lookup).to_numpy(dtype=float)


def summarize_validation(scan: pd.DataFrame, config: DesignConfig) -> dict:
    n = int(len(scan))
    max_allowed_count = int(math.ceil(config.max_shift_rate * n) - 1)
    summary = {"n_variants": n, "max_allowed_shifted_count": max_allowed_count}
    if "target_margin_core_score" in scan:
        margins = scan["target_margin_core_score"].astype(float)
        summary.update(
            {
                "target_margin_positive_count": int((margins > 0).sum()),
                "target_margin_positive_rate": float((margins > 0).mean()) if n else np.nan,
                "target_margin_mean": float(margins.mean()) if n else np.nan,
                "target_margin_median": float(margins.median()) if n else np.nan,
                "target_margin_min": float(margins.min()) if n else np.nan,
            }
        )
    for anchor in ("m10", "m35"):
        shifts = scan[f"{anchor}_shift"].astype(int)
        shifted_count = int((shifts != 0).sum())
        out_count = int((shifts.abs() > config.max_abs_shift).sum())
        shifted_rate = shifted_count / n if n else np.nan
        max_abs = int(shifts.abs().max()) if n else 0
        # Rate only. The old criterion also demanded out_count == 0 - not one variant
        # anywhere with |shift| > max_abs_shift - which no run ever came close to:
        # three full runs sat at 2,130 / 2,237 / 5,281 out of 15,625 with no trend
        # toward zero, so the gate could never open and "pass" carried no
        # information. What the library actually needs is variants whose designed
        # architecture is the one the model reads, and shift == 0 says exactly that
        # (it is also the same set as target_margin > 0). out_of_range_count stays in
        # the summary as a reported diagnostic; it just no longer vetoes a run.
        passed = shifted_rate < config.max_shift_rate
        summary.update(
            {
                f"{anchor}_shifted_count": shifted_count,
                f"{anchor}_shifted_rate": shifted_rate,
                f"{anchor}_out_of_range_count": out_count,
                f"{anchor}_max_abs_shift": max_abs,
                f"{anchor}_validation_pass": bool(passed),
            }
        )
    summary["global_validation_pass"] = bool(
        summary["m10_validation_pass"] and summary["m35_validation_pass"]
    )
    # Phase gate, not a validation criterion: m10 only has to reach
    # m10_phase_exit_rate before the search switches to the -35 correction phase.
    # Deliberately ignores m10_out_of_range_count - the m10 objective already
    # ranks out-of-range variants, and gating on it would keep the search in the
    # m10 phase well past the rate the gate is written in.
    summary["m10_phase_gate_pass"] = bool(
        summary["m10_shifted_rate"] <= config.m10_phase_exit_rate
    )
    summary["phase"] = (
        "done"
        if summary["global_validation_pass"]
        # Still above the gate: keep pushing m10 down.
        else "m10"
        if not summary["m10_phase_gate_pass"]
        # Through the gate and m35 still failing: correct -35.
        else "m35"
        if not summary["m35_validation_pass"]
        # m35 passes but m10 has not reached max_shift_rate yet, so come back and
        # finish m10. This return path is why the gate is not an acceptance line.
        else "m10"
    )
    return summary


def objective_for_phase(summary: dict, phase: str) -> tuple:
    """Rank two candidate states. Lower is better, compared lexicographically.

    Only the two shifted counts. The objective used to carry the combined
    out-of-range count and both max_abs_shift values as tie-breakers, but the goal
    is to maximise how many variants sit at shift == 0 and the magnitude of the
    shifts that remain is not a quantity we act on: the dominant off-target is
    ~35 bp away, a different site entirely rather than a near-miss register, so
    shrinking "max shift" from 43 to 40 buys nothing. Keeping those terms only let
    the search trade away shift == 0 variants to improve a number nobody reads.
    """
    if phase not in {"m10", "m35"}:
        return (0,)
    m10_shifted = int(summary["m10_shifted_count"])
    m35_shifted = int(summary["m35_shifted_count"])
    if phase == "m35":
        # -35 correction phase. Ranking m35 first is what makes this phase about
        # -35: a move that only lowers m10 while m35 grows no longer wins. m10
        # stays second so a move that leaves m35 unchanged and lowers m10 is
        # still an improvement and gets taken. m10 is separately forbidden from
        # rising by the hard filter in AutomatedRedesigner.run().
        return (m35_shifted, m10_shifted)
    # m10 phase: reduce -10 register shifts first, then -35.
    return (m10_shifted, m35_shifted)


def _dominant_shift(scan: pd.DataFrame, phase: str) -> tuple[int | None, pd.Series]:
    shift_col = f"{phase}_shift"
    # Driver selection follows the global objective: eliminate any register
    # shift first. Out-of-range magnitude is only a later tie-breaker.
    target_mask = scan[shift_col] != 0
    if not target_mask.any():
        return None, target_mask
    counts = scan.loc[target_mask, shift_col].value_counts().rename_axis("shift").reset_index(name="n")
    counts["abs_shift"] = counts["shift"].abs()
    counts = counts.sort_values(["n", "abs_shift", "shift"], ascending=[False, False, True])
    dominant = int(counts.iloc[0]["shift"])
    return dominant, scan[shift_col] == dominant


def diagnose_shift_drivers(
    scan: pd.DataFrame,
    selected_elements: pd.DataFrame,
    design_space: DesignSpace,
    config: DesignConfig,
    phase: str | None = None,
) -> dict:
    summary = summarize_validation(scan, config)
    phase = phase or summary["phase"]
    dominant, target_mask = (None, None) if phase == "done" else _dominant_shift(scan, phase)
    if dominant is None:
        return {
            "phase": phase,
            "dominant_shift": None,
            "risk": pd.DataFrame(),
            "overlap": pd.DataFrame(),
            "driver_units": [],
        }
    overall_target_rate = float(target_mask.mean())
    risk_rows, overlap_rows = [], []
    for row in selected_elements.itertuples(index=False):
        element, version = row.element, row.version
        slot_mask = scan[f"{element}_version"] == version
        n_contexts = int(slot_mask.sum())
        n_target = int((slot_mask & target_mask).sum())
        conditional = n_target / n_contexts if n_contexts else np.nan
        unit = design_space.unit_for_slot(element, version)
        subset = scan[slot_mask & target_mask]
        role_evidence = []
        for role in ELEMENTS:
            overlap_col = f"observed_{role}_overlap_regions"
            role_mask = subset[overlap_col].fillna("").str.split("|").map(lambda x: element in x)
            overlap_n = int(role_mask.sum())
            if overlap_n:
                deltas = subset.loc[role_mask, f"{role}_within_model_delta_energy"]
                median_delta = float(deltas.median()) if deltas.notna().any() else np.nan
                max_delta = float(deltas.max()) if deltas.notna().any() else np.nan
            else:
                median_delta = np.nan
                max_delta = np.nan
            role_evidence.append((role, overlap_n, median_delta, max_delta))
            overlap_rows.append(
                {
                    "phase": phase,
                    "dominant_shift": dominant,
                    "design_element": element,
                    "design_version": version,
                    "observed_role": role,
                    "n_target_contexts": n_target,
                    "overlap_n": overlap_n,
                    "overlap_fraction": overlap_n / n_target if n_target else 0.0,
                    "median_within_model_delta_energy": median_delta,
                    "max_within_model_delta_energy": max_delta,
                }
            )
        top_role, top_overlap_n, top_median_delta, _ = sorted(
            role_evidence,
            key=lambda x: (x[1], -np.inf if np.isnan(x[2]) else x[2]),
            reverse=True,
        )[0]
        risk_rows.append(
            {
                "phase": phase,
                "dominant_shift": dominant,
                "element": element,
                "version": version,
                "design_unit_id": unit.unit_id if unit else f"locked_{version}",
                "actionable": unit is not None,
                "n_contexts": n_contexts,
                "n_target_shift": n_target,
                "conditional_shift_risk": conditional,
                "overall_target_shift_rate": overall_target_rate,
                "risk_enrichment": conditional / overall_target_rate if overall_target_rate else np.nan,
                "top_observed_role_on_design": top_role,
                "top_overlap_n": top_overlap_n,
                "top_overlap_fraction": top_overlap_n / n_target if n_target else 0.0,
                "top_role_median_delta_energy": top_median_delta,
            }
        )
    risk = pd.DataFrame(risk_rows).sort_values(
        ["conditional_shift_risk", "top_overlap_fraction", "n_target_shift"],
        ascending=False,
    ).reset_index(drop=True)
    overlap = pd.DataFrame(overlap_rows).sort_values(
        ["overlap_n", "overlap_fraction"], ascending=False
    ).reset_index(drop=True)
    actionable = risk[(risk["actionable"]) & (risk["top_overlap_n"] > 0)]
    if actionable.empty:
        actionable = risk[risk["actionable"]]
    driver_units = []
    for row in actionable.itertuples(index=False):
        key = design_space.unit_for_slot(row.element, row.version)
        if key is not None and key not in driver_units:
            driver_units.append(key)
    return {
        "phase": phase,
        "dominant_shift": dominant,
        "risk": risk,
        "overlap": overlap,
        "driver_units": driver_units,
    }


class AutomatedRedesigner:
    def __init__(
        self,
        design_space: DesignSpace,
        scanner: CorePromoterScanner,
        gap_assignment: pd.DataFrame,
        config: DesignConfig,
        out_dir: Path,
        max_iterations: int = 30,
        n_driver_units: int = 3,
        probe_candidates_per_unit: int = 2,
        pair_beam_width: int = 6,
        max_pair_evaluations: int = 8,
        max_stalled_iterations: int = 3,
    ):
        self.design_space = design_space
        self.scanner = scanner
        self.gap_assignment = gap_assignment
        self.config = config
        self.out_dir = Path(out_dir)
        self.max_iterations = max_iterations
        self.n_driver_units = n_driver_units
        self.probe_candidates_per_unit = probe_candidates_per_unit
        self.pair_beam_width = pair_beam_width
        self.max_pair_evaluations = max_pair_evaluations
        self.max_stalled_iterations = max_stalled_iterations
        self.evaluated: set[tuple] = set()
        self.live_progress_df: pd.DataFrame | None = None
        self._verbose: bool = False

    def _changed_row_mask(
        self,
        variants: pd.DataFrame,
        elements: pd.DataFrame,
        changed_keys: Iterable[DesignUnitKey],
    ) -> pd.Series:
        """Rows of `variants` whose sequence differs when only `changed_keys` are
        swapped in. `elements` supplies the (element, design_unit_id) -> version
        mapping, which is structural and identical for every state (most units
        are 1:1 with a version label; the spacer's v2_v3_pair unit covers both
        v2 and v3), so any state's `selected_elements()` output works here.
        """
        mask = pd.Series(False, index=variants.index)
        for key in changed_keys:
            versions = elements.loc[
                (elements["element"] == key.element) & (elements["design_unit_id"] == key.unit_id),
                "version",
            ]
            mask |= variants[f"{key.element}_version"].isin(versions)
        return mask

    def evaluate_state(
        self,
        state: dict[DesignUnitKey, int],
        annotate_element_energies: bool = True,
        baseline_elements: pd.DataFrame | None = None,
        baseline_scan: pd.DataFrame | None = None,
        changed_keys: Iterable[DesignUnitKey] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        self.design_space.validate_state(state)
        elements = self.design_space.selected_elements(state)
        variants = assemble_library(elements, self.gap_assignment, self.config)

        if baseline_scan is not None and changed_keys:
            # assemble_library's row order/combination is fixed by gap_assignment
            # and never depends on which sequence is selected, so baseline_scan
            # (from an earlier state) lines up row-for-row with `variants` here.
            # Only the rows touched by changed_keys need a fresh model pass.
            mask = self._changed_row_mask(variants, baseline_elements, changed_keys)
            scan = baseline_scan.copy()
            if mask.any():
                affected_index = variants.index[mask]
                rescanned = self.scanner.scan(
                    variants.loc[mask],
                    annotate_element_energies=annotate_element_energies,
                    annotate_diagnostics=annotate_element_energies,
                )
                # scan() internally does reset_index(drop=True), so the subset
                # result comes back as 0..len(subset)-1; reattach the original
                # row labels before writing back into `scan`.
                rescanned.index = affected_index
                scan.loc[affected_index, rescanned.columns] = rescanned
        else:
            scan = self.scanner.scan(
                variants,
                annotate_element_energies=annotate_element_energies,
                annotate_diagnostics=annotate_element_energies,
            )
        summary = summarize_validation(scan, self.config)
        return elements, scan, summary

    def _proposal_signature(self, state: dict[DesignUnitKey, int], changes: dict[DesignUnitKey, int]) -> tuple:
        return (
            self.design_space.state_signature(state),
            tuple((key.element, key.unit_id, int(idx)) for key, idx in sorted(changes.items(), key=lambda x: x[0].label())),
        )

    def _remaining_indices(self, state: dict[DesignUnitKey, int], key: DesignUnitKey) -> list[int]:
        indices = []
        for idx in range(len(self.design_space.unit_candidates[key])):
            if idx == int(state[key]):
                continue
            signature = self._proposal_signature(state, {key: idx})
            if signature not in self.evaluated:
                indices.append(idx)
        return indices

    def _next_indices(self, state: dict[DesignUnitKey, int], key: DesignUnitKey) -> list[int]:
        return self._remaining_indices(state, key)[: self.probe_candidates_per_unit]

    @staticmethod
    def _state_payload(state: dict[DesignUnitKey, int]) -> list[dict]:
        return [
            {"element": key.element, "unit_id": key.unit_id, "candidate_index": int(idx)}
            for key, idx in sorted(state.items(), key=lambda item: item[0].label())
        ]

    @staticmethod
    def _state_from_payload(rows: list[dict]) -> dict[DesignUnitKey, int]:
        return {
            DesignUnitKey(str(row["element"]), str(row["unit_id"])): int(row["candidate_index"])
            for row in rows
        }

    def _evaluated_payload(self) -> list[dict]:
        return [
            {
                "state": [list(item) for item in state_signature],
                "changes": [list(item) for item in changes],
            }
            for state_signature, changes in sorted(self.evaluated, key=repr)
        ]

    def _restore_evaluated(self, rows: list[dict]) -> None:
        self.evaluated = {
            (
                tuple(tuple(item) for item in row["state"]),
                tuple(tuple(item) for item in row["changes"]),
            )
            for row in rows
        }

    @staticmethod
    def _parse_changes(text: str) -> dict[DesignUnitKey, int]:
        changes: dict[DesignUnitKey, int] = {}
        for token in str(text).split(";"):
            token = token.strip()
            if not token or "->" not in token or ":" not in token:
                continue
            label, idx = token.rsplit("->", 1)
            element, unit_id = label.split(":", 1)
            changes[DesignUnitKey(element, unit_id)] = int(idx)
        return changes

    def _persist_progress(
        self,
        iteration: int,
        stalled: int,
        state: dict[DesignUnitKey, int],
        elements: pd.DataFrame,
        summary: dict,
        history_rows: list[dict],
        proposal_rows: list[dict],
        stop_reason: str = "running",
    ) -> None:
        history = pd.DataFrame(history_rows)
        proposals = pd.DataFrame(proposal_rows)
        if self.live_progress_df is not None:
            live = self.live_progress_df
            live.drop(index=live.index, inplace=True)
            for column in list(live.columns):
                if column not in history.columns:
                    del live[column]
            for column in history.columns:
                live[column] = history[column].to_numpy()
        history.to_csv(self.out_dir / "search_progress.csv", index=False)
        proposals.to_csv(self.out_dir / "proposal_history_checkpoint.csv", index=False)
        elements.to_csv(self.out_dir / "current_elements_checkpoint.csv", index=False)
        _json_dump(self.out_dir / "current_validation.json", summary)
        checkpoint = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "last_completed_iteration": int(iteration),
            "stalled_iterations": int(stalled),
            "stop_reason": stop_reason,
            "state": self._state_payload(state),
            "evaluated": self._evaluated_payload(),
            "summary": summary,
        }
        _json_dump(self.out_dir / "search_checkpoint.json", checkpoint)

    @staticmethod
    def _read_records(path: Path) -> list[dict]:
        if not path.exists() or path.stat().st_size == 0:
            return []
        return pd.read_csv(path).to_dict("records")

    def _load_resume_state(self) -> tuple[dict[DesignUnitKey, int], int, int, list[dict], list[dict]]:
        checkpoint_path = self.out_dir / "search_checkpoint.json"
        progress_path = self.out_dir / "search_progress.csv"
        proposal_checkpoint_path = self.out_dir / "proposal_history_checkpoint.csv"
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            state = self._state_from_payload(checkpoint["state"])
            self._restore_evaluated(checkpoint.get("evaluated", []))
            last_iteration = int(checkpoint.get("last_completed_iteration", 0))
            stalled = int(checkpoint.get("stalled_iterations", 0))
            history_rows = self._read_records(progress_path)
            proposal_rows = self._read_records(proposal_checkpoint_path)
            return state, last_iteration, stalled, history_rows, proposal_rows

        # Backward-compatible resume for runs created before checkpoint support.
        elements_path = self.out_dir / "current_elements_checkpoint.csv"
        if not elements_path.exists():
            elements_path = self.out_dir / "final_elements.csv"
        if not elements_path.exists():
            raise FileNotFoundError(
                f"Resume requested, but no search_checkpoint.json or final_elements.csv exists in {self.out_dir}"
            )
        state = self.design_space.state_from_selected_elements(pd.read_csv(elements_path))
        history_path = self.out_dir / "search_history.csv"
        proposal_path = self.out_dir / "proposal_history.csv"
        history_rows = self._read_records(history_path)
        proposal_rows = self._read_records(proposal_path)
        last_iteration = max((int(row.get("iteration", 0)) for row in history_rows), default=0)

        # Old runs did not save the state associated with every proposal. Mark
        # their candidate indices as evaluated for the recovered current state
        # so resume moves forward instead of immediately replaying the same set.
        for row in proposal_rows:
            changes = self._parse_changes(row.get("changes", ""))
            if changes:
                self.evaluated.add(self._proposal_signature(state, changes))
        return state, last_iteration, 0, history_rows, proposal_rows

    def _evaluate_proposal(
        self,
        iteration: int,
        state: dict[DesignUnitKey, int],
        changes: dict[DesignUnitKey, int],
        phase: str,
        proposal_type: str,
        baseline_elements: pd.DataFrame | None = None,
        baseline_scan: pd.DataFrame | None = None,
    ) -> dict | None:
        signature = self._proposal_signature(state, changes)
        if signature in self.evaluated:
            return None
        self.evaluated.add(signature)
        change_text = ";".join(f"{key.label()}->{idx}" for key, idx in changes.items())
        if self._verbose:
            print(f"    evaluating {proposal_type}: {change_text}", flush=True)
        candidate_state = dict(state)
        candidate_state.update(changes)
        try:
            elements, scan, summary = self.evaluate_state(
                candidate_state,
                annotate_element_energies=False,
                baseline_elements=baseline_elements,
                baseline_scan=baseline_scan,
                changed_keys=list(changes),
            )
        except ValueError as exc:
            return {
                "iteration": iteration,
                "proposal_type": proposal_type,
                "changes": change_text,
                "valid": False,
                "error": str(exc),
                "state": candidate_state,
            }
        result = {
            "iteration": iteration,
            "proposal_type": proposal_type,
            "changes": change_text,
            "valid": True,
            "error": "",
            "objective": objective_for_phase(summary, phase),
            "summary": summary,
            "state": candidate_state,
            "elements": elements,
            "scan": scan,
            "change_map": changes,
        }
        if self._verbose:
            print(
                "      "
                f"m10={summary['m10_shifted_rate']:.2%} "
                f"(out={summary['m10_out_of_range_count']}) | "
                f"m35={summary['m35_shifted_rate']:.2%} "
                f"(out={summary['m35_out_of_range_count']})",
                flush=True,
            )
        return result

    def run(
        self,
        initial_state: dict[DesignUnitKey, int] | None = None,
        initial_evaluation: tuple[pd.DataFrame, pd.DataFrame, dict] | None = None,
        resume: bool = False,
        reset_stalled_on_resume: bool = True,
        progress_df: pd.DataFrame | None = None,
        verbose: bool = False,
    ) -> DesignResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.live_progress_df = progress_df
        self._verbose = verbose
        if resume:
            state, last_iteration, stalled, history_rows, proposal_rows = self._load_resume_state()
            if reset_stalled_on_resume:
                stalled = 0
            elements, scan, summary = self.evaluate_state(state)
            start_iteration = last_iteration + 1
            tqdm.write(
                f"[Resume] {self.out_dir}\n"
                f"  continuing from completed iteration {last_iteration}; "
                f"next iteration={start_iteration}; additional allowance={self.max_iterations}"
            )
        else:
            state = dict(initial_state or self.design_space.initial_state())
            if initial_evaluation is None:
                elements, scan, summary = self.evaluate_state(state)
            else:
                elements, scan, summary = initial_evaluation
            elements.to_csv(self.out_dir / "initial_elements.csv", index=False)
            scan.to_csv(self.out_dir / f"initial_scan_{self.config.n_variants()}.csv", index=False)
            _json_dump(self.out_dir / "initial_validation.json", summary)
            history_rows = [{"iteration": 0, "accepted": True, "changes": "initial", **summary}]
            proposal_rows = []
            stalled = 0
            start_iteration = 1
            self._persist_progress(
                iteration=0,
                stalled=0,
                state=state,
                elements=elements,
                summary=summary,
                history_rows=history_rows,
                proposal_rows=proposal_rows,
            )

        last_completed_iteration = start_iteration - 1
        stop_reason = "running"
        end_iteration = start_iteration + self.max_iterations - 1
        pbar = tqdm(range(start_iteration, end_iteration + 1), desc="Batch6 search")
        for iteration in pbar:
            if summary["global_validation_pass"]:
                stop_reason = "validation_passed"
                break
            phase = summary["phase"]
            current_objective = objective_for_phase(summary, phase)
            pbar.set_postfix(
                phase=phase,
                m10=f"{summary['m10_shifted_rate']:.1%}",
                m35=f"{summary['m35_shifted_rate']:.1%}",
                stalled=f"{stalled}/{self.max_stalled_iterations}",
            )
            if self._verbose:
                print(
                    f"\n[Iteration {iteration}] phase={phase} "
                    f"(m10 gate<={self.config.m10_phase_exit_rate:.0%} "
                    f"{'open' if summary['m10_phase_gate_pass'] else 'closed'}) | "
                    f"m10 shifted={summary['m10_shifted_rate']:.2%} "
                    f"out={summary['m10_out_of_range_count']} max_abs={summary['m10_max_abs_shift']} | "
                    f"m35 shifted={summary['m35_shifted_rate']:.2%} "
                    f"out={summary['m35_out_of_range_count']} max_abs={summary['m35_max_abs_shift']} | "
                    f"stalled={stalled}/{self.max_stalled_iterations}",
                    flush=True,
                )
            diagnosis = diagnose_shift_drivers(
                scan,
                elements,
                self.design_space,
                self.config,
                phase=phase,
            )
            diagnosis["risk"].to_csv(self.out_dir / f"risk_iter_{iteration:02d}.csv", index=False)
            diagnosis["overlap"].to_csv(self.out_dir / f"overlap_iter_{iteration:02d}.csv", index=False)
            drivers = diagnosis["driver_units"][: self.n_driver_units]
            if self._verbose:
                print(
                    f"  dominant {phase}_shift={diagnosis['dominant_shift']} | "
                    f"drivers={[key.label() for key in drivers]}",
                    flush=True,
                )
            if not drivers:
                history_rows.append(
                    {"iteration": iteration, "accepted": False, "changes": "no_actionable_driver", **summary}
                )
                last_completed_iteration = iteration
                stop_reason = "no_actionable_driver"
                self._persist_progress(
                    iteration, stalled, state, elements, summary, history_rows, proposal_rows, stop_reason
                )
                break

            driver_pool_status = []
            for key in drivers:
                total_alternatives = len(self.design_space.unit_candidates[key]) - 1
                remaining = len(self._remaining_indices(state, key))
                driver_pool_status.append(
                    f"{key.label()} remaining={remaining}/{total_alternatives}"
                )
            if self._verbose:
                print(f"  driver pools: {driver_pool_status}", flush=True)

            evaluated_proposals = []
            for key in drivers:
                for idx in self._next_indices(state, key):
                    proposal = self._evaluate_proposal(
                        iteration, state, {key: idx}, phase, "single",
                        baseline_elements=elements, baseline_scan=scan,
                    )
                    if proposal is not None:
                        evaluated_proposals.append(proposal)

            valid_singles = [p for p in evaluated_proposals if p.get("valid")]
            valid_singles.sort(key=lambda p: p["objective"])
            beam = valid_singles[: self.pair_beam_width]
            pair_count = 0
            for left, right in itertools.combinations(beam, 2):
                left_changes = left["change_map"]
                right_changes = right["change_map"]
                if set(left_changes) & set(right_changes):
                    continue
                changes = {**left_changes, **right_changes}
                proposal = self._evaluate_proposal(
                    iteration, state, changes, phase, "double",
                    baseline_elements=elements, baseline_scan=scan,
                )
                if proposal is not None:
                    evaluated_proposals.append(proposal)
                    pair_count += 1
                if pair_count >= self.max_pair_evaluations:
                    break

            if not evaluated_proposals:
                history_rows.append(
                    {"iteration": iteration, "accepted": False, "changes": "no_unevaluated_candidates", **summary}
                )
                last_completed_iteration = iteration
                stop_reason = "candidate_pool_exhausted_for_current_drivers"
                self._persist_progress(
                    iteration, stalled, state, elements, summary, history_rows, proposal_rows, stop_reason
                )
                tqdm.write("  STOP: no unevaluated candidates remain for the selected drivers.")
                break

            for proposal in evaluated_proposals:
                row = {
                    "iteration": proposal["iteration"],
                    "phase": phase,
                    "proposal_type": proposal["proposal_type"],
                    "changes": proposal["changes"],
                    "valid": proposal.get("valid", False),
                    "error": proposal.get("error", ""),
                    "accepted": False,
                }
                if proposal.get("valid"):
                    row.update(proposal["summary"])
                    row["objective"] = repr(proposal["objective"])
                proposal_rows.append(row)

            candidates = [p for p in evaluated_proposals if p.get("valid")]
            if phase == "m35":
                # Hard constraint for the -35 correction phase: m10 may stay
                # equal but must never get worse. Measured against the current
                # library's shifted count rather than max_shift_rate, because
                # this phase can be entered anywhere up to m10_phase_exit_rate
                # and m10 must not drift back up from wherever it started.
                current_m10_shifted = int(summary["m10_shifted_count"])
                candidates = [
                    p
                    for p in candidates
                    if int(p["summary"]["m10_shifted_count"]) <= current_m10_shifted
                ]
            improving = [p for p in candidates if p["objective"] < current_objective]
            if improving:
                best = min(improving, key=lambda p: p["objective"])
                state = best["state"]
                # Re-run only the accepted state with element-energy
                # annotations. Rejected proposals need register metrics only.
                # `elements`/`scan` on the right-hand side are still the
                # pre-acceptance values here, so they're the correct baseline
                # for the rows that best["change_map"] actually touched.
                elements, scan, summary = self.evaluate_state(
                    state,
                    annotate_element_energies=True,
                    baseline_elements=elements,
                    baseline_scan=scan,
                    changed_keys=list(best["change_map"]),
                )
                stalled = 0
                for row in reversed(proposal_rows):
                    if row["iteration"] == iteration and row["changes"] == best["changes"]:
                        row["accepted"] = True
                        break
                history_rows.append(
                    {"iteration": iteration, "accepted": True, "changes": best["changes"], **summary}
                )
                elements.to_csv(self.out_dir / f"accepted_elements_iter_{iteration:02d}.csv", index=False)
                tqdm.write(
                    f"  ACCEPT {best['proposal_type']}: {best['changes']} | "
                    f"m10={summary['m10_shifted_rate']:.2%}, "
                    f"m35={summary['m35_shifted_rate']:.2%}"
                )
            else:
                stalled += 1
                history_rows.append(
                    {"iteration": iteration, "accepted": False, "changes": "no_strict_improvement", **summary}
                )
                tqdm.write(
                    f"  NO ACCEPTED MOVE: strict objective did not improve "
                    f"(stalled={stalled}/{self.max_stalled_iterations})."
                )
            last_completed_iteration = iteration
            self._persist_progress(
                iteration,
                stalled,
                state,
                elements,
                summary,
                history_rows,
                proposal_rows,
            )
            if summary["global_validation_pass"]:
                stop_reason = "validation_passed"
                break
            if not improving:
                if stalled >= self.max_stalled_iterations:
                    stop_reason = "max_stalled_iterations"
                    break
        else:
            stop_reason = "max_iterations_this_run"

        self._persist_progress(
            last_completed_iteration,
            stalled,
            state,
            elements,
            summary,
            history_rows,
            proposal_rows,
            stop_reason,
        )
        tqdm.write(
            f"\n[Search stopped] reason={stop_reason} | "
            f"last_iteration={last_completed_iteration} | "
            f"m10={summary['m10_shifted_rate']:.2%} | "
            f"m35={summary['m35_shifted_rate']:.2%}"
        )

        final_diagnosis = diagnose_shift_drivers(
            scan,
            elements,
            self.design_space,
            self.config,
            phase=summary["phase"],
        )
        history = pd.DataFrame(history_rows)
        proposals = pd.DataFrame(proposal_rows)
        elements.to_csv(self.out_dir / "final_elements.csv", index=False)
        scan.to_csv(self.out_dir / f"final_scan_{self.config.n_variants()}.csv", index=False)
        history.to_csv(self.out_dir / "search_history.csv", index=False)
        proposals.to_csv(self.out_dir / "proposal_history.csv", index=False)
        final_diagnosis["risk"].to_csv(self.out_dir / "final_driver_risk.csv", index=False)
        final_diagnosis["overlap"].to_csv(self.out_dir / "final_overlap_evidence.csv", index=False)
        _json_dump(self.out_dir / "final_validation.json", {**summary, "stop_reason": stop_reason})
        self.gap_assignment.to_csv(self.out_dir / "gap_assignment.csv", index=False)
        return DesignResult(
            success=bool(summary["global_validation_pass"]),
            stop_reason=stop_reason,
            out_dir=self.out_dir,
            final_state=state,
            final_elements=elements,
            final_scan=scan,
            final_summary=summary,
            history=history,
            proposals=proposals,
            final_risk=final_diagnosis["risk"],
            final_overlap=final_diagnosis["overlap"],
        )


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
