# -*- coding: utf-8 -*-
"""The monotonic automated redesign search.

Entry point in 01_recursive_design.ipynb: Batch 6.
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from ._common import _json_dump
from .config import DesignConfig, DesignResult, DesignUnitKey
from .space import DesignSpace, assemble_library
from .scanning import CorePromoterScanner, diagnose_shift_drivers, objective_for_phase, summarize_validation


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
