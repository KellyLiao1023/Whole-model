#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PAS / BPM library QC pipeline
=============================

Scan a bacterial promoter combinatorial library with the repository's
pretrained PAS / BPM model (``scripts/BPM/BPM.py`` + ``Params_Con17.pkl``)
to check whether a *fixed-register* promoter design is reliable, and whether
the PAS-assigned (best-scoring) architectures are diverse enough for
downstream / shadow-promoter analysis.

This script does **not** retrain PAS. It only *uses* the frozen parameters.

Scoring convention (verified empirically against BPM.py)
--------------------------------------------------------
``BPM.score_promoter(core)`` returns a promoter free energy ``dG`` where
**lower energy == stronger promoter** (consensus TTGACA / TATAAT are the
global minima). Therefore:

    * "best" architecture  == argmin(E)
    * delta_E = E_best_off - E_design  ->  delta_E < 0  means an off-register
      architecture is *stronger* than the designed one (a QC risk).
    * Boltzmann weight       W_i = exp(-(E_i - E_min))   (E_min for stability)
    * P_design = W_design / (W_design + sum W_off)

If you swap in a model whose score is higher-is-better, set
``--score_is_energy false`` and the code negates internally so that all of the
above semantics are preserved.

BPM only scores the *core* promoter (-35 [6] + spacer + -10 [6]). It does not
model UP / Dis / ITS. E_design is therefore a core-promoter energy. Hooks for
an external element-score table are provided but never required.

Author: generated for the promoter-library-exp repository.
"""

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt

# tqdm is optional
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# logomaker is optional (sequence logos)
try:
    import logomaker
    _HAVE_LOGOMAKER = True
except Exception:  # pragma: no cover
    _HAVE_LOGOMAKER = False


# --------------------------------------------------------------------------- #
# BPM / PAS model import
# --------------------------------------------------------------------------- #
def import_bpm(scripts_dir: str):
    """Import the repository BPM module (``scripts/BPM/BPM.py``).

    We add ``scripts_dir`` to ``sys.path`` so that ``import BPM.BPM`` resolves
    ``BPM/__init``-less package via the folder. BPM.py itself does
    ``from BPM.util import *`` so the parent (scripts) dir must be importable.
    """
    scripts_dir = os.path.abspath(scripts_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        import BPM.BPM as bpm  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            f"Could not import BPM from {scripts_dir!r}. "
            f"Expected {scripts_dir}/BPM/BPM.py . Original error: {exc}"
        )
    # Sanity: required callables
    for fn in ("score_promoter", "score_m35", "score_m10", "score_spacer"):
        if not hasattr(bpm, fn):
            raise ImportError(f"BPM module missing required function {fn!r}")
    return bpm


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
VALID_BASES = set("ACGT")


@dataclass
class LayoutConfig:
    """All lengths are configurable; nothing is hard-coded downstream.

    Region order along the full sequence:
        5'RE | UP | -35 | spacer | -10 | Dis | ITS | internal_RE | barcode | 3'RE
    The promoter (69 nt by default) = UP + -35 + spacer + -10 + Dis + ITS.
    """
    re5_len: int = 0
    up_len: int = 22
    minus35_len: int = 6
    spacer_len_design: int = 17
    minus10_len: int = 6
    dis_len: int = 8
    its_len: int = 10
    internal_re_len: int = 12
    barcode_len: int = 15
    re3_len: int = 0

    spacer_lengths_allowed: Tuple[int, ...] = (15, 16, 17, 18, 19)

    @property
    def promoter_len(self) -> int:
        return (self.up_len + self.minus35_len + self.spacer_len_design
                + self.minus10_len + self.dis_len + self.its_len)

    # ---- promoter-relative coordinates ----
    @property
    def prom_up_start(self) -> int:
        return 0

    @property
    def prom_minus35_start(self) -> int:
        return self.up_len

    @property
    def prom_spacer_start(self) -> int:
        return self.prom_minus35_start + self.minus35_len

    @property
    def prom_minus10_start(self) -> int:
        return self.prom_spacer_start + self.spacer_len_design

    @property
    def prom_dis_start(self) -> int:
        return self.prom_minus10_start + self.minus10_len

    @property
    def prom_its_start(self) -> int:
        return self.prom_dis_start + self.dis_len

    # ---- full-sequence coordinates (offsets by re5 + promoter position) ----
    @property
    def full_promoter_start(self) -> int:
        return self.re5_len

    @property
    def designed_minus35_start_full(self) -> int:
        return self.re5_len + self.prom_minus35_start

    @property
    def designed_minus10_start_full(self) -> int:
        return self.re5_len + self.prom_minus10_start

    def region_intervals(self) -> List[Tuple[str, int, int]]:
        """Return list of (region_name, start, end) in full-sequence coords."""
        cur = 0
        out = []
        for name, length in [
            ("5RE", self.re5_len),
            ("UP", self.up_len),
            ("designed_-35", self.minus35_len),
            ("designed_spacer", self.spacer_len_design),
            ("designed_-10", self.minus10_len),
            ("Dis", self.dis_len),
            ("ITS", self.its_len),
            ("internal_RE", self.internal_re_len),
            ("barcode", self.barcode_len),
            ("3RE", self.re3_len),
        ]:
            out.append((name, cur, cur + length))
            cur += length
        return out

    @property
    def expected_full_len(self) -> int:
        return sum(length for _, s, e in self.region_intervals() for length in [e - s])


@dataclass
class Thresholds:
    p_design_clean: float = 0.8
    p_design_high_risk: float = 0.5
    delta_margin: float = 0.0
    off_register_high_risk_target: float = 0.10  # want < 10%


# --------------------------------------------------------------------------- #
# Sequence helpers
# --------------------------------------------------------------------------- #
def is_valid_dna(seq: str) -> bool:
    return len(seq) > 0 and set(seq.upper()) <= VALID_BASES


def region_for_interval(layout: LayoutConfig, start: int, end: int) -> Tuple[str, List[str]]:
    """Classify a [start, end) motif window into a source region.

    Returns (region_label, overlap_list). If the window spans >1 region the
    label is 'mixed' and overlap_list holds every region touched.
    """
    touched = []
    for name, rs, re_ in layout.region_intervals():
        if re_ <= rs:
            continue  # zero-length region
        if start < re_ and end > rs:  # overlap
            touched.append(name)
    if not touched:
        return "out", []
    if len(touched) == 1:
        return touched[0], touched
    return "mixed", touched


# --------------------------------------------------------------------------- #
# Candidate enumeration + scoring
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    minus35_start_full: int
    spacer_len: int
    minus35_seq: str
    spacer_seq: str
    minus10_seq: str
    minus10_start_full: int
    candidate_core_seq: str
    energy: float
    is_designed_register: bool
    source_region_minus35: str
    source_region_minus10: str
    overlap_internal_RE: bool
    overlap_barcode: bool
    overlap_flanking_RE: bool


def enumerate_candidates(full_seq: str, layout: LayoutConfig, bpm,
                         score_is_energy: bool) -> List[Candidate]:
    """Enumerate every -35 / spacer / -10 architecture over the full sequence.

    candidate = 6 bp -35 + spacer(15..19) + 6 bp -10, contiguous.
    Energy uses BPM.score_promoter on the concatenated core (m35 + spacer + m10).
    If score_is_energy is False (higher-is-better model), we negate so lower==better.
    """
    n = len(full_seq)
    m35_len = layout.minus35_len
    m10_len = layout.minus10_len
    designed_m35 = layout.designed_minus35_start_full
    designed_m10 = layout.designed_minus10_start_full

    cands: List[Candidate] = []
    for sp_len in layout.spacer_lengths_allowed:
        block = m35_len + sp_len + m10_len
        for m35_start in range(0, n - block + 1):
            sp_start = m35_start + m35_len
            m10_start = sp_start + sp_len
            m35_seq = full_seq[m35_start:sp_start]
            sp_seq = full_seq[sp_start:m10_start]
            m10_seq = full_seq[m10_start:m10_start + m10_len]
            core = m35_seq + sp_seq + m10_seq

            # Guard: non-ACGT hexamers score 0.0 silently in BPM -> skip.
            if not (is_valid_dna(m35_seq) and is_valid_dna(m10_seq)):
                continue

            raw = bpm.score_promoter(core)
            energy = raw if score_is_energy else -raw

            is_design = (m35_start == designed_m35 and m10_start == designed_m10
                         and sp_len == layout.spacer_len_design)

            reg35, ov35 = region_for_interval(layout, m35_start, sp_start)
            reg10, ov10 = region_for_interval(layout, m10_start, m10_start + m10_len)
            touched = set(ov35) | set(ov10)
            cands.append(Candidate(
                minus35_start_full=m35_start,
                spacer_len=sp_len,
                minus35_seq=m35_seq,
                spacer_seq=sp_seq,
                minus10_seq=m10_seq,
                minus10_start_full=m10_start,
                candidate_core_seq=core,
                energy=float(energy),
                is_designed_register=is_design,
                source_region_minus35=reg35,
                source_region_minus10=reg10,
                overlap_internal_RE=("internal_RE" in touched),
                overlap_barcode=("barcode" in touched),
                overlap_flanking_RE=(("5RE" in touched) or ("3RE" in touched)),
            ))
    return cands


# --------------------------------------------------------------------------- #
# Per-variant reduction
# --------------------------------------------------------------------------- #
def boltzmann_dominance(energies: np.ndarray, is_design: np.ndarray
                        ) -> Tuple[float, float, float]:
    """Return (P_design, W_design, W_off_total) with overflow-safe weights.

    Lower energy == better. W_i = exp(-(E_i - E_min)).
    """
    e_min = float(np.min(energies))
    w = np.exp(-(energies - e_min))  # in (0, 1], stable
    w_design = float(w[is_design].sum())
    w_off = float(w[~is_design].sum())
    denom = w_design + w_off
    p_design = w_design / denom if denom > 0 else float("nan")
    return p_design, w_design, w_off


def summarize_variant(vid, barcode, full_seq, promoter, layout: LayoutConfig,
                      bpm, score_is_energy: bool, thr: Thresholds
                      ) -> Tuple[dict, List[dict]]:
    """Score one variant. Returns (per_variant_row, long_candidate_rows)."""
    # Designed core straight from promoter coordinates
    d_m35 = promoter[layout.prom_minus35_start:layout.prom_minus35_start + layout.minus35_len]
    d_sp = promoter[layout.prom_spacer_start:layout.prom_spacer_start + layout.spacer_len_design]
    d_m10 = promoter[layout.prom_minus10_start:layout.prom_minus10_start + layout.minus10_len]
    designed_core = d_m35 + d_sp + d_m10

    row = {
        "variant_id": vid,
        "barcode": barcode,
        "full_sequence": full_seq,
        "promoter_sequence": promoter,
        "designed_minus35_seq": d_m35,
        "designed_spacer_seq": d_sp,
        "designed_minus10_seq": d_m10,
    }

    # Validity gate
    if not is_valid_dna(full_seq):
        row.update(_nan_scoring_fields(reason="invalid_bases"))
        return row, []

    raw_design = bpm.score_promoter(designed_core) if is_valid_dna(designed_core) else np.nan
    e_design = raw_design if score_is_energy else -raw_design

    cands = enumerate_candidates(full_seq, layout, bpm, score_is_energy)
    if not cands:
        row.update(_nan_scoring_fields(reason="no_candidates"))
        return row, []

    energies = np.array([c.energy for c in cands], dtype=float)
    is_design_arr = np.array([c.is_designed_register for c in cands], dtype=bool)

    # If the designed register was not enumerable (edge geometry), fall back to
    # the computed designed-core energy for E_design but keep candidate stats.
    if is_design_arr.any():
        e_design_from_scan = float(energies[is_design_arr].min())
    else:
        e_design_from_scan = float(e_design) if np.isfinite(e_design) else float("nan")

    # For a consistent Boltzmann ensemble, ensure the designed energy is in the
    # pool. Build an augmented pool (candidates + guaranteed designed entry).
    if is_design_arr.any():
        pool_e = energies
        pool_isd = is_design_arr
    else:
        pool_e = np.append(energies, e_design_from_scan)
        pool_isd = np.append(is_design_arr, True)

    p_design, w_design, w_off = boltzmann_dominance(pool_e, pool_isd)

    e_best_all = float(energies.min())
    best_idx = int(np.argmin(energies))
    best_cand = cands[best_idx]

    off_mask = ~is_design_arr
    if off_mask.any():
        off_energies = energies[off_mask]
        off_best_local = int(np.argmin(off_energies))
        off_cands = [c for c, m in zip(cands, off_mask) if m]
        best_off = off_cands[off_best_local]
        e_best_off = float(best_off.energy)
    else:
        best_off = None
        e_best_off = float("nan")

    best_all_is_design = bool(best_cand.is_designed_register)
    delta_e = (e_best_off - e_design_from_scan) if np.isfinite(e_best_off) else float("nan")

    minus35_shift = best_cand.minus35_start_full - layout.designed_minus35_start_full
    minus10_shift = best_cand.minus10_start_full - layout.designed_minus10_start_full

    # hard-max label: is the single best architecture the designed register?
    hardmax_label = "hardmax_design" if best_all_is_design else "hardmax_shifted"

    # classification
    classification = classify(p_design, best_all_is_design, delta_e, thr)

    row.update({
        "E_design": e_design_from_scan,
        "E_best_all": e_best_all,
        "E_best_off": e_best_off,
        "delta_E": delta_e,
        "P_design": p_design,
        "W_design": w_design,
        "W_off_total": w_off,
        "best_all_is_design": best_all_is_design,
        "classification": classification,
        "hardmax_shift_label": hardmax_label,
        "best_minus35_start_full": best_cand.minus35_start_full,
        "best_minus10_start_full": best_cand.minus10_start_full,
        "minus35_shift": minus35_shift,
        "minus10_shift": minus10_shift,
        "best_minus35_seq": best_cand.minus35_seq,
        "best_minus10_seq": best_cand.minus10_seq,
        "best_spacer_len": best_cand.spacer_len,
        "best_minus35_source_region": best_cand.source_region_minus35,
        "best_minus10_source_region": best_cand.source_region_minus10,
        "best_off_minus35_seq": best_off.minus35_seq if best_off else "",
        "best_off_minus10_seq": best_off.minus10_seq if best_off else "",
        "best_off_spacer_len": best_off.spacer_len if best_off else np.nan,
        "best_off_minus35_source_region": best_off.source_region_minus35 if best_off else "",
        "best_off_minus10_source_region": best_off.source_region_minus10 if best_off else "",
    })

    # long-format candidate rows (only emit designed + best_all + best_off to
    # keep the file manageable; full enumeration is available on request).
    long_rows = []
    for c in cands:
        roles = []
        if c.is_designed_register:
            roles.append("designed")
        if id(c) == id(best_cand):
            roles.append("best_all")
        if best_off is not None and id(c) == id(best_off):
            roles.append("best_off")
        if not roles:
            continue
        long_rows.append({
            "variant_id": vid, "candidate_role": "+".join(roles),
            "minus35_start_full": c.minus35_start_full,
            "minus35_seq": c.minus35_seq, "spacer_len": c.spacer_len,
            "spacer_seq": c.spacer_seq,
            "minus10_start_full": c.minus10_start_full, "minus10_seq": c.minus10_seq,
            "candidate_core_seq": c.candidate_core_seq, "energy": c.energy,
            "is_designed_register": c.is_designed_register,
            "source_region_minus35": c.source_region_minus35,
            "source_region_minus10": c.source_region_minus10,
            "overlap_internal_RE": c.overlap_internal_RE,
            "overlap_barcode": c.overlap_barcode,
            "overlap_flanking_RE": c.overlap_flanking_RE,
        })
    return row, long_rows


def _nan_scoring_fields(reason: str) -> dict:
    keys = ["E_design", "E_best_all", "E_best_off", "delta_E", "P_design",
            "W_design", "W_off_total", "minus35_shift", "minus10_shift",
            "best_minus35_start_full", "best_minus10_start_full", "best_spacer_len",
            "best_off_spacer_len"]
    d = {k: np.nan for k in keys}
    d.update({
        "best_all_is_design": False,
        "classification": f"skipped:{reason}",
        "hardmax_shift_label": "skipped",
        "best_minus35_seq": "", "best_minus10_seq": "",
        "best_minus35_source_region": "", "best_minus10_source_region": "",
        "best_off_minus35_seq": "", "best_off_minus10_seq": "",
        "best_off_minus35_source_region": "", "best_off_minus10_source_region": "",
    })
    return d


def classify(p_design, best_all_is_design, delta_e, thr: Thresholds) -> str:
    if not np.isfinite(p_design):
        return "skipped:nan"
    high_risk = (p_design <= thr.p_design_high_risk) or \
                ((not best_all_is_design) and np.isfinite(delta_e)
                 and delta_e < thr.delta_margin)
    if high_risk:
        return "off_register_high_risk"
    if p_design >= thr.p_design_clean:
        return "clean_fixed_register"
    return "ambiguous"


# --------------------------------------------------------------------------- #
# Diversity / entropy helpers
# --------------------------------------------------------------------------- #
def positionwise_base_freq(motifs: List[str], k: int) -> pd.DataFrame:
    counts = np.zeros((k, 4))
    idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    n = 0
    for m in motifs:
        if len(m) != k or not is_valid_dna(m):
            continue
        n += 1
        for i, b in enumerate(m):
            counts[i, idx[b]] += 1
    freq = counts / n if n else counts
    df = pd.DataFrame(freq, columns=["A", "C", "G", "T"])
    df.index.name = "position"
    return df


def shannon_entropy_per_position(freq_df: pd.DataFrame) -> List[float]:
    ent = []
    for _, r in freq_df.iterrows():
        p = r.values
        p = p[p > 0]
        ent.append(float(-(p * np.log2(p)).sum()))
    return ent


def motif_diversity_table(motifs: List[str], k: int, label: str) -> dict:
    ser = pd.Series([m for m in motifs if len(m) == k and is_valid_dna(m)])
    n = len(ser)
    vc = ser.value_counts()
    top10 = vc.head(10)
    top1_freq = float(vc.iloc[0] / n) if n else float("nan")
    top5_freq = float(vc.head(5).sum() / n) if n else float("nan")
    freq_df = positionwise_base_freq(list(ser), k)
    ent = shannon_entropy_per_position(freq_df)
    return {
        "label": label,
        "n": n,
        "n_unique": int(ser.nunique()),
        "top1_freq": top1_freq,
        "top5_freq": top5_freq,
        "top10": top10,
        "posfreq": freq_df,
        "entropy": ent,
    }


# --------------------------------------------------------------------------- #
# Full-sequence assembly
# --------------------------------------------------------------------------- #
def assemble_full_sequence(promoter, barcode, re5, internal_re, re3) -> str:
    return f"{re5}{promoter}{internal_re}{barcode}{re3}"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(df: pd.DataFrame, fig_dir: str, thr: Thresholds,
                 div10_all: dict, div35_all: dict,
                 src10: pd.Series, src35: pd.Series):
    os.makedirs(fig_dir, exist_ok=True)
    scored = df[df["classification"].str.startswith("skipped") == False].copy()

    def hist(col, title, fname, vline=None):
        vals = pd.to_numeric(scored[col], errors="coerce").dropna()
        if vals.empty:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals, bins=40, color="#4c72b0", edgecolor="black", lw=0.4)
        if vline is not None:
            ax.axvline(vline, color="red", lw=1)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("variants")
        _savefig(fig, os.path.join(fig_dir, fname))

    hist("E_design", "Designed-register energy", "01_E_design_hist.png")
    hist("P_design", "Designed-register dominance P_design", "02_P_design_hist.png",
         vline=thr.p_design_clean)
    hist("delta_E", "delta_E = E_best_off - E_design", "03_delta_E_hist.png", vline=0.0)
    hist("minus10_shift", "-10 hard-max shift distance", "04_minus10_shift_hist.png", vline=0)
    hist("minus35_shift", "-35 hard-max shift distance", "05_minus35_shift_hist.png", vline=0)

    # stacked barplot of variant classes
    order = ["clean_fixed_register", "ambiguous", "off_register_high_risk"]
    counts = scored["classification"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = 0
    colors = {"clean_fixed_register": "#55a868", "ambiguous": "#dd8452",
              "off_register_high_risk": "#c44e52"}
    for cls in order:
        ax.bar(["library"], [counts[cls]], bottom=bottom, label=cls, color=colors[cls])
        bottom += counts[cls]
    ax.set_ylabel("variants")
    ax.set_title("Variant classification")
    ax.legend(fontsize=8)
    _savefig(fig, os.path.join(fig_dir, "06_variant_classes.png"))

    # source-region distribution of predicted -10 / -35 (best_all)
    for src, name, fname in [(src10, "predicted -10", "07_source_region_minus10.png"),
                             (src35, "predicted -35", "08_source_region_minus35.png")]:
        if src is None or src.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        src.plot(kind="bar", ax=ax, color="#8172b3")
        ax.set_title(f"Source region of {name} (best_all)")
        ax.set_ylabel("variants")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        _savefig(fig, os.path.join(fig_dir, fname))

    # top predicted motif frequency barplots
    for div, name, fname in [(div10_all, "-10", "09_top_minus10_motifs.png"),
                             (div35_all, "-35", "10_top_minus35_motifs.png")]:
        top10 = div.get("top10")
        if top10 is None or len(top10) == 0:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        top10.plot(kind="bar", ax=ax, color="#4c72b0")
        ax.set_title(f"Top predicted {name} hexamers (best_all)")
        ax.set_ylabel("count")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        _savefig(fig, os.path.join(fig_dir, fname))

    # optional logos
    if _HAVE_LOGOMAKER:
        for div, name, fname in [(div10_all, "-10", "11_logo_minus10.png"),
                                 (div35_all, "-35", "11_logo_minus35.png")]:
            freq = div.get("posfreq")
            if freq is None or freq.empty:
                continue
            try:
                fig, ax = plt.subplots(figsize=(4, 2))
                logomaker.Logo(freq, ax=ax)
                ax.set_title(f"Predicted {name} logo")
                _savefig(fig, os.path.join(fig_dir, fname))
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Energy bins
# --------------------------------------------------------------------------- #
def energy_bins_summary(e: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    e = pd.to_numeric(e, errors="coerce").dropna()
    if e.empty:
        return pd.DataFrame()
    cats = pd.cut(e, bins=n_bins)
    tab = cats.value_counts().sort_index()
    df = tab.rename_axis("energy_bin").reset_index(name="count")
    df["fraction"] = df["count"] / df["count"].sum()
    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PAS/BPM fixed-register library QC pipeline (uses pretrained params).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant_table", required=True, help="CSV/TSV of variants.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--scripts_dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="Directory containing BPM/ package.")

    p.add_argument("--sequence_column", default="full_sequence",
                   help="Column holding full sequence OR promoter (see --sequence_is).")
    p.add_argument("--sequence_is", choices=["full", "promoter"], default="full",
                   help="Whether --sequence_column holds full sequence or 69nt promoter.")
    p.add_argument("--promoter_column", default=None,
                   help="Explicit promoter column (optional; overrides).")
    p.add_argument("--barcode_column", default="barcode")
    p.add_argument("--id_column", default="variant_id")

    # flanks for building full sequence from promoter (+barcode)
    p.add_argument("--re5_sequence", default="")
    p.add_argument("--internal_re_sequence", default="")
    p.add_argument("--re3_sequence", default="")

    # layout lengths
    p.add_argument("--up_len", type=int, default=22)
    p.add_argument("--minus35_len", type=int, default=6)
    p.add_argument("--spacer_len_design", type=int, default=17)
    p.add_argument("--minus10_len", type=int, default=6)
    p.add_argument("--dis_len", type=int, default=8)
    p.add_argument("--its_len", type=int, default=10)
    p.add_argument("--internal_re_len", type=int, default=12)
    p.add_argument("--barcode_len", type=int, default=15)
    p.add_argument("--spacer_lengths_allowed", default="15,16,17,18,19")

    # scoring convention
    p.add_argument("--score_is_energy", default="true",
                   choices=["true", "false"],
                   help="true: model score is energy (lower=better, BPM default).")

    # thresholds
    p.add_argument("--p_design_clean", type=float, default=0.8)
    p.add_argument("--p_design_high_risk", type=float, default=0.5)
    p.add_argument("--delta_margin", type=float, default=0.0)

    # optional element-score table hook
    p.add_argument("--element_score_table", default=None,
                   help="Optional CSV: element_name,element_id,energy")
    p.add_argument("--limit", type=int, default=0, help="Debug: only first N variants (0=all).")
    return p


def read_table(path: str) -> pd.DataFrame:
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def main(argv=None):
    args = build_argparser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")

    score_is_energy = (args.score_is_energy == "true")
    bpm = import_bpm(args.scripts_dir)

    re5 = args.re5_sequence.upper()
    internal_re = args.internal_re_sequence.upper()
    re3 = args.re3_sequence.upper()

    layout = LayoutConfig(
        re5_len=len(re5),
        up_len=args.up_len,
        minus35_len=args.minus35_len,
        spacer_len_design=args.spacer_len_design,
        minus10_len=args.minus10_len,
        dis_len=args.dis_len,
        its_len=args.its_len,
        internal_re_len=len(internal_re) if internal_re else args.internal_re_len,
        barcode_len=args.barcode_len,
        re3_len=len(re3),
        spacer_lengths_allowed=tuple(int(x) for x in args.spacer_lengths_allowed.split(",")),
    )
    thr = Thresholds(args.p_design_clean, args.p_design_high_risk, args.delta_margin)

    print(f"[config] promoter_len={layout.promoter_len} "
          f"designed -35@full={layout.designed_minus35_start_full} "
          f"-10@full={layout.designed_minus10_start_full}")
    print(f"[config] score_is_energy={score_is_energy} "
          f"(lower={'better' if score_is_energy else 'worse'})")

    df = read_table(args.variant_table)
    if args.limit:
        df = df.head(args.limit)
    print(f"[input] {len(df)} variants from {args.variant_table}")

    seq_col = args.sequence_column
    prom_col = args.promoter_column
    id_col = args.id_column
    bc_col = args.barcode_column

    # Build per-variant rows
    per_variant = []
    long_rows = []
    n_skipped = 0
    for i, (_, r) in enumerate(tqdm(df.iterrows(), total=len(df), desc="scoring")):
        vid = r[id_col] if id_col in df.columns else f"var_{i:06d}"
        barcode = str(r[bc_col]).upper() if bc_col in df.columns and pd.notna(r.get(bc_col)) else ""

        # resolve promoter + full sequence
        if prom_col and prom_col in df.columns:
            promoter = str(r[prom_col]).upper()
        elif args.sequence_is == "promoter":
            promoter = str(r[seq_col]).upper()
        else:
            promoter = None

        if args.sequence_is == "full":
            full_seq = str(r[seq_col]).upper()
            if promoter is None:
                # extract promoter from full using coordinates
                ps = layout.full_promoter_start
                promoter = full_seq[ps:ps + layout.promoter_len]
        else:
            # build full from promoter + barcode + flanks
            full_seq = assemble_full_sequence(promoter, barcode, re5, internal_re, re3)

        row, lrows = summarize_variant(vid, barcode, full_seq, promoter, layout,
                                       bpm, score_is_energy, thr)
        # carry through element id columns if present
        for col in ("UP_id", "minus35_id", "spacer_id", "minus10_id", "Dis_id", "ITS_id",
                    "UP", "m35", "spacer", "m10", "DIS", "ITS"):
            if col in df.columns:
                row[col] = r[col]
        per_variant.append(row)
        long_rows.extend(lrows)
        if str(row["classification"]).startswith("skipped"):
            n_skipped += 1

    pv = pd.DataFrame(per_variant)
    long_df = pd.DataFrame(long_rows)

    # ------------------------------------------------------------------ #
    # Summaries
    # ------------------------------------------------------------------ #
    scored = pv[~pv["classification"].str.startswith("skipped")].copy()
    total = len(pv)
    n_scored = len(scored)

    class_counts = scored["classification"].value_counts()
    clean = int(class_counts.get("clean_fixed_register", 0))
    ambig = int(class_counts.get("ambiguous", 0))
    highrisk = int(class_counts.get("off_register_high_risk", 0))
    hardmax_shifted = int((scored["hardmax_shift_label"] == "hardmax_shifted").sum())

    pct = lambda x: (100.0 * x / n_scored) if n_scored else float("nan")
    highrisk_pct = pct(highrisk)
    qc_pass = highrisk_pct < (thr.off_register_high_risk_target * 100)

    # ---- energy-space diversity (A) ----
    e_design = scored["E_design"]
    e_desc = {
        "min": float(e_design.min()), "max": float(e_design.max()),
        "mean": float(e_design.mean()), "median": float(e_design.median()),
        "std": float(e_design.std()),
        "q05": float(e_design.quantile(0.05)), "q25": float(e_design.quantile(0.25)),
        "q50": float(e_design.quantile(0.50)), "q75": float(e_design.quantile(0.75)),
        "q95": float(e_design.quantile(0.95)),
    }
    ebins = energy_bins_summary(e_design, n_bins=10)
    single_bin_dominant = bool((ebins["fraction"] > 0.40).any()) if not ebins.empty else False
    empty_bins = int((ebins["count"] == 0).sum()) if not ebins.empty else 0

    # ---- PAS-assigned motif diversity (C) ----
    def motif_group(mask, k, col):
        return motif_diversity_table(list(scored.loc[mask, col].dropna()), k, col)

    groups = {
        "all": scored.index,
        "clean_fixed_register": scored.index[scored["classification"] == "clean_fixed_register"],
        "ambiguous": scored.index[scored["classification"] == "ambiguous"],
        "off_register_high_risk": scored.index[scored["classification"] == "off_register_high_risk"],
        "hardmax_shifted": scored.index[scored["hardmax_shift_label"] == "hardmax_shifted"],
    }
    # motifs overlapping internal_RE / barcode (best_all -10 or -35 region)
    art_mask = scored["best_minus10_source_region"].isin(["internal_RE", "barcode"]) | \
               scored["best_minus35_source_region"].isin(["internal_RE", "barcode"])
    groups["overlap_RE_or_barcode"] = scored.index[art_mask]

    div10_rows, div35_rows = [], []
    div10_all = div35_all = None
    for gname, gidx in groups.items():
        d10 = motif_diversity_table(list(scored.loc[gidx, "best_minus10_seq"].dropna()),
                                    layout.minus10_len, f"{gname}:-10")
        d35 = motif_diversity_table(list(scored.loc[gidx, "best_minus35_seq"].dropna()),
                                    layout.minus35_len, f"{gname}:-35")
        if gname == "all":
            div10_all, div35_all = d10, d35
        div10_rows.append(_flatten_div(gname, d10))
        div35_rows.append(_flatten_div(gname, d35))

    motif10_df = pd.DataFrame(div10_rows)
    motif35_df = pd.DataFrame(div35_rows)

    low_div_10 = (div10_all["top1_freq"] > 0.15) or (div10_all["top5_freq"] > 0.50)
    low_div_35 = (div35_all["top1_freq"] > 0.15) or (div35_all["top5_freq"] > 0.50)

    # ---- source-region analysis (E) ----
    # best_off regions
    src_off10 = scored["best_off_minus10_source_region"].replace("", np.nan).dropna()
    src_off35 = scored["best_off_minus35_source_region"].replace("", np.nan).dropna()
    src_best10 = scored["best_minus10_source_region"].value_counts()
    src_best35 = scored["best_minus35_source_region"].value_counts()

    def pct_region(series, region):
        return float((series == region).mean() * 100) if len(series) else float("nan")

    source_region_summary = pd.DataFrame({
        "region": sorted(set(src_off10.unique()) | set(src_off35.unique())),
    })
    source_region_summary["best_off_minus10_count"] = source_region_summary["region"].map(
        src_off10.value_counts()).fillna(0).astype(int)
    source_region_summary["best_off_minus35_count"] = source_region_summary["region"].map(
        src_off35.value_counts()).fillna(0).astype(int)

    artifact = {
        "best_off_minus10_in_internal_RE_pct": pct_region(src_off10, "internal_RE"),
        "best_off_minus10_in_barcode_pct": pct_region(src_off10, "barcode"),
        "best_off_minus35_in_internal_RE_pct": pct_region(src_off35, "internal_RE"),
        "best_off_minus35_in_barcode_pct": pct_region(src_off35, "barcode"),
    }
    artifact_risk = any(v > 20 for v in artifact.values() if np.isfinite(v))

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #
    make_figures(pv, fig_dir, thr, div10_all, div35_all, src_best10, src_best35)

    # ------------------------------------------------------------------ #
    # Write outputs
    # ------------------------------------------------------------------ #
    od = args.output_dir
    pv.to_csv(os.path.join(od, "pas_qc_per_variant.csv"), index=False)
    long_df.to_csv(os.path.join(od, "pas_qc_candidates_long.csv"), index=False)
    motif10_df.to_csv(os.path.join(od, "motif_diversity_minus10.csv"), index=False)
    motif35_df.to_csv(os.path.join(od, "motif_diversity_minus35.csv"), index=False)
    source_region_summary.to_csv(os.path.join(od, "source_region_summary.csv"), index=False)
    ebins.to_csv(os.path.join(od, "energy_bins_summary.csv"), index=False)

    summary = {
        "input_table": args.variant_table,
        "total_variants": total,
        "scored_variants": n_scored,
        "skipped_variants": n_skipped,
        "hardmax_shift_rate_pct": pct(hardmax_shifted),
        "clean_fixed_register": {"count": clean, "pct": pct(clean)},
        "ambiguous": {"count": ambig, "pct": pct(ambig)},
        "off_register_high_risk": {"count": highrisk, "pct": highrisk_pct},
        "qc_off_register_target_pct": thr.off_register_high_risk_target * 100,
        "qc_pass_off_register": bool(qc_pass),
        "E_design_stats": e_desc,
        "energy_single_bin_over_40pct": single_bin_dominant,
        "energy_empty_bins": empty_bins,
        "motif_minus10_low_diversity_flag": bool(low_div_10),
        "motif_minus35_low_diversity_flag": bool(low_div_35),
        "motif_minus10_top1_freq": div10_all["top1_freq"],
        "motif_minus10_top5_freq": div10_all["top5_freq"],
        "motif_minus35_top1_freq": div35_all["top1_freq"],
        "motif_minus35_top5_freq": div35_all["top5_freq"],
        "artifact_region_pct": artifact,
        "artifact_risk_flag": bool(artifact_risk),
        "score_is_energy": score_is_energy,
        "layout": {k: getattr(layout, k) for k in
                   ["re5_len", "up_len", "minus35_len", "spacer_len_design",
                    "minus10_len", "dis_len", "its_len", "internal_re_len",
                    "barcode_len", "re3_len", "promoter_len"]},
    }
    with open(os.path.join(od, "pas_qc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    write_summary_md(os.path.join(od, "pas_qc_summary.md"), summary, thr)

    # ------------------------------------------------------------------ #
    # Console report
    # ------------------------------------------------------------------ #
    print("\n==================== PAS LIBRARY QC ====================")
    print(f"total variants          : {total}")
    print(f"scored / skipped        : {n_scored} / {n_skipped}")
    print(f"hard-max shift rate      : {pct(hardmax_shifted):.1f}%")
    print(f"clean_fixed_register     : {clean} ({pct(clean):.1f}%)")
    print(f"ambiguous                : {ambig} ({pct(ambig):.1f}%)")
    print(f"off_register_high_risk   : {highrisk} ({highrisk_pct:.1f}%)  "
          f"[target < {thr.off_register_high_risk_target*100:.0f}%]  "
          f"{'PASS' if qc_pass else 'FAIL'}")
    print(f"E_design range           : [{e_desc['min']:.2f}, {e_desc['max']:.2f}] "
          f"median {e_desc['median']:.2f}")
    print(f"-10 motif top1/top5      : {div10_all['top1_freq']:.2f} / {div10_all['top5_freq']:.2f}"
          f"  {'LOW-DIV!' if low_div_10 else ''}")
    print(f"-35 motif top1/top5      : {div35_all['top1_freq']:.2f} / {div35_all['top5_freq']:.2f}"
          f"  {'LOW-DIV!' if low_div_35 else ''}")
    print(f"artifact (RE/barcode)    : {'RISK' if artifact_risk else 'ok'}  {artifact}")
    print(f"outputs -> {od}")
    print("========================================================")


def _flatten_div(gname, d):
    return {
        "group": gname, "n": d["n"], "n_unique": d["n_unique"],
        "top1_freq": d["top1_freq"], "top5_freq": d["top5_freq"],
        "top10_motifs": ";".join(f"{k}:{v}" for k, v in d["top10"].items()),
        "entropy_per_pos": ";".join(f"{e:.3f}" for e in d["entropy"]),
    }


def write_summary_md(path, s, thr: Thresholds):
    L = s["layout"]
    lines = []
    A = lines.append
    A("# PAS / BPM Library QC Summary\n")
    A(f"- Input: `{s['input_table']}`")
    A(f"- Total variants: **{s['total_variants']}** "
      f"(scored {s['scored_variants']}, skipped {s['skipped_variants']})")
    A(f"- Scoring convention: `score_is_energy = {s['score_is_energy']}` "
      f"({'lower energy = stronger promoter' if s['score_is_energy'] else 'higher score = stronger; negated internally'})")
    A("")
    A("## Layout (configurable)\n")
    A(f"5'RE {L['re5_len']} | UP {L['up_len']} | -35 {L['minus35_len']} | "
      f"spacer {L['spacer_len_design']} | -10 {L['minus10_len']} | Dis {L['dis_len']} | "
      f"ITS {L['its_len']} | internal_RE {L['internal_re_len']} | "
      f"barcode {L['barcode_len']} | 3'RE {L['re3_len']}  "
      f"(promoter = {L['promoter_len']} nt)\n")

    A("## Fixed-register QC\n")
    A(f"- Hard-max shift rate: **{s['hardmax_shift_rate_pct']:.1f}%**")
    A(f"- clean_fixed_register: **{s['clean_fixed_register']['count']} "
      f"({s['clean_fixed_register']['pct']:.1f}%)**  (P_design >= {thr.p_design_clean})")
    A(f"- ambiguous: {s['ambiguous']['count']} ({s['ambiguous']['pct']:.1f}%)  "
      f"({thr.p_design_high_risk} < P_design < {thr.p_design_clean})")
    A(f"- off_register_high_risk: **{s['off_register_high_risk']['count']} "
      f"({s['off_register_high_risk']['pct']:.1f}%)**")
    A(f"- **QC target off_register_high_risk < {s['qc_off_register_target_pct']:.0f}%: "
      f"{'PASS ✅' if s['qc_pass_off_register'] else 'FAIL ❌'}**\n")

    A("## Energy-space coverage (E_design)\n")
    e = s["E_design_stats"]
    A(f"- min/median/max: {e['min']:.2f} / {e['median']:.2f} / {e['max']:.2f}; "
      f"mean {e['mean']:.2f}, std {e['std']:.2f}")
    A(f"- quantiles 5/25/50/75/95: {e['q05']:.2f} / {e['q25']:.2f} / "
      f"{e['q50']:.2f} / {e['q75']:.2f} / {e['q95']:.2f}")
    A(f"- single energy bin > 40% of variants: "
      f"{'YES ⚠️' if s['energy_single_bin_over_40pct'] else 'no'}; "
      f"empty bins: {s['energy_empty_bins']}")
    A("- Note: a full-factorial energy distribution need not be uniform; we only "
      "require low / medium / high energy regions to each hold enough variants.\n")

    A("## PAS-assigned motif diversity\n")
    A(f"- predicted -10 top1 / top5: {s['motif_minus10_top1_freq']:.2f} / "
      f"{s['motif_minus10_top5_freq']:.2f}  "
      f"{'⚠️ low-diversity / consensus-artifact risk' if s['motif_minus10_low_diversity_flag'] else ''}")
    A(f"- predicted -35 top1 / top5: {s['motif_minus35_top1_freq']:.2f} / "
      f"{s['motif_minus35_top5_freq']:.2f}  "
      f"{'⚠️ low-diversity / consensus-artifact risk' if s['motif_minus35_low_diversity_flag'] else ''}")
    A("  (flag if top1 > 0.15 or top5 > 0.50)\n")

    A("## Artifact source-region check (best off-register motifs)\n")
    for k, v in s["artifact_region_pct"].items():
        A(f"- {k}: {v:.1f}%")
    A(f"- **artifact risk (RE/barcode capture): "
      f"{'YES ⚠️' if s['artifact_risk_flag'] else 'no'}**\n")

    A("## Interpretation notes (important)\n")
    A("1. PAS hard-max *shifted* does **not** mean RNAP actually uses a shifted "
      "promoter; it is a model prediction.")
    A("2. PAS is model-based architecture assignment, **not** experimental TSS mapping.")
    A("3. `P_design` is a model-based measure of designed-register dominance.")
    A("4. high-risk variants mean the fixed-register interpretation may not be clean.")
    A("5. PAS-assigned motif diversity analysis is **data QC**, not biological validation.")
    A("6. If off_register_high_risk < 10%, E_design spans low/medium/high, and PAS-assigned "
      "motifs are not dominated by a few hexamers or by RE/barcode artifacts, then this "
      "library is better suited to downstream fixed-register and shadow-promoter analysis.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
