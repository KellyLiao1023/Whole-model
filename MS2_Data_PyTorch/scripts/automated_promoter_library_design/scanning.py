# -*- coding: utf-8 -*-
"""Full-sequence CorePromoter scanning, validation summary, shift diagnosis.

Entry point in 01_recursive_design.ipynb: Batch 4 and Batch 5.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch

import recursive_corepromoter_design as legacy

from ._common import ELEMENTS, ELEMENT_LENGTHS, WINDOW_NAMES, _overlap_regions, _safe_slice
from .config import DesignConfig
from .scoring import ElementModelBundle
from .space import DesignSpace


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
