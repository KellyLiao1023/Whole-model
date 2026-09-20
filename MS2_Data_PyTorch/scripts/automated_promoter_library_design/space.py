# -*- coding: utf-8 -*-
"""The design space: units, candidate selection, gap assignment, library assembly.

Entry point in 01_recursive_design.ipynb: Batch 2 and Batch 3.
"""
from __future__ import annotations

import itertools
import random

import numpy as np
import pandas as pd

from ._common import ELEMENTS, ELEMENT_LENGTHS, SPACER_DERIVED_VERSION_IDX, normalize_dna, pool_bin_edges, version_index
from .config import DesignConfig, DesignUnitKey
from .scoring import ElementModelBundle, binding_locked_energy, locked_energies


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
