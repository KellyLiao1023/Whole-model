from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from recursive_corepromoter_design import CHANNEL_BY_SPACER, PROJECT_ROOT, SEED, TABLE_DIR


DEFAULT_XLSX = TABLE_DIR / "Data_S1_20250826.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tss_pas_dataset"
SEQUENCE_COLUMNS = ("UP", "Minus35", "Spacer", "Minus10", "Dis", "Start", "ITR")
REQUIRED_COLUMNS = ("TSS", "Direction", *SEQUENCE_COLUMNS)
HEADER_ALIASES = {
    "TSS": ("TSS", "TSS_coordinate"),
    "Direction": ("Direction",),
    "Minus10_3end_coordinate": ("Minus10_3'", "Minus10_3end_coordinate"),
    "UP": ("UP",),
    "Minus35": ("Minus35",),
    "Spacer": ("Spacer",),
    "Minus10": ("Minus10",),
    "Dis": ("Dis",),
    "Start": ("Start",),
    "ITR": ("ITR",),
}
VALID_BASES = frozenset("ACGTN")
MODEL_INPUT_LENGTH = 80
MODEL_RECEPTIVE_FIELD = 72
MODEL_N_POSITIONS = MODEL_INPUT_LENGTH - MODEL_RECEPTIVE_FIELD + 1
MODEL_M35_OFFSET = 22


@dataclass(frozen=True)
class TSSDataConfig:
    xlsx_path: str
    seed: int = SEED
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    allow_n: bool = True


def normalize_header(value: object) -> str:
    return str(value).strip()


def normalize_sequence(value: object, *, empty_dash: bool = False) -> str:
    if pd.isna(value):
        return ""
    seq = "".join(str(value).upper().split()).replace("U", "T")
    if empty_dash and seq and set(seq) == {"-"}:
        return ""
    return seq


def _find_source_column(columns: Iterable[object], aliases: Iterable[str]) -> object | None:
    normalized = {normalize_header(column): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def standardize_sheet(raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    source_columns = {
        canonical: _find_source_column(raw.columns, aliases)
        for canonical, aliases in HEADER_ALIASES.items()
    }
    missing_headers = [column for column in REQUIRED_COLUMNS if source_columns.get(column) is None]
    if missing_headers:
        raise ValueError(f"Sheet {sheet_name!r} is missing required columns: {missing_headers}")

    out = pd.DataFrame(index=raw.index)
    for canonical, source in source_columns.items():
        out[canonical] = raw[source] if source is not None else pd.NA
    out["source_sheet"] = sheet_name
    # Sheets such as My.sm and My.sm$ are different sources for the same organism.
    out["species_group"] = sheet_name.rstrip("$")
    out["source_row"] = np.arange(2, len(out) + 2, dtype=int)
    return out.reset_index(drop=True)


def read_tss_workbook(xlsx_path: Path | str = DEFAULT_XLSX) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)
    sheets = pd.read_excel(xlsx_path, sheet_name=None, dtype=object)
    standardized = [standardize_sheet(raw, name) for name, raw in sheets.items()]
    if not standardized:
        raise ValueError(f"Workbook contains no sheets: {xlsx_path}")
    return pd.concat(standardized, ignore_index=True)


def _sequence_qc_reason(row: pd.Series, allow_n: bool) -> str:
    reasons: list[str] = []
    allowed = VALID_BASES if allow_n else frozenset("ACGT")
    expected_lengths = {"UP": 100, "Minus35": 6, "Minus10": 6, "Start": 3, "ITR": 100}
    for column, expected in expected_lengths.items():
        seq = row[column]
        if len(seq) != expected:
            reasons.append(f"{column}_length_{len(seq)}")
    spacer_length = len(row["Spacer"])
    if spacer_length not in CHANNEL_BY_SPACER:
        reasons.append(f"unsupported_spacer_{spacer_length}")
    for column in SEQUENCE_COLUMNS:
        invalid = sorted(set(row[column]) - allowed)
        if invalid:
            reasons.append(f"{column}_invalid_{''.join(invalid)}")
    if str(row["Direction"]).strip() not in {"+", "-"}:
        reasons.append("invalid_direction")
    return "|".join(reasons)


def clean_tss_dataframe(raw: pd.DataFrame, *, allow_n: bool = True) -> pd.DataFrame:
    out = raw.copy()
    for column in SEQUENCE_COLUMNS:
        out[column] = out[column].map(
            lambda value, col=column: normalize_sequence(value, empty_dash=(col == "Dis"))
        )
    out["Direction"] = out["Direction"].astype("string").str.strip()
    out["spacer_length"] = out["Spacer"].str.len().astype(int)
    out["design_channel"] = out["spacer_length"].map(CHANNEL_BY_SPACER)
    out["qc_reason"] = out.apply(_sequence_qc_reason, axis=1, allow_n=allow_n)
    out["qc_pass"] = out["qc_reason"].eq("")
    out["m35_start"] = out["UP"].str.len().astype(int)
    out["full_sequence"] = out[list(SEQUENCE_COLUMNS)].agg("".join, axis=1)
    out["full_sequence_length"] = out["full_sequence"].str.len().astype(int)
    return out


def assign_grouped_splits(
    dataframe: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = SEED,
) -> pd.DataFrame:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation_fraction and test_fraction must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")

    out = dataframe.copy()
    groups = out["species_group"].astype(str)
    first = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_idx, test_idx = next(first.split(out, groups=groups))
    train_val = out.iloc[train_val_idx]
    relative_validation = validation_fraction / (1.0 - test_fraction)
    second = GroupShuffleSplit(n_splits=1, test_size=relative_validation, random_state=seed + 1)
    train_rel_idx, validation_rel_idx = next(
        second.split(train_val, groups=train_val["species_group"].astype(str))
    )

    split = pd.Series(index=out.index, dtype="string")
    split.iloc[test_idx] = "test"
    split.iloc[train_val_idx[train_rel_idx]] = "train"
    split.iloc[train_val_idx[validation_rel_idx]] = "validation"
    if split.isna().any():
        raise RuntimeError("Grouped split left rows unassigned")
    out["split"] = split
    return out


def build_tss_pas_dataframe(
    xlsx_path: Path | str = DEFAULT_XLSX,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = SEED,
    allow_n: bool = True,
) -> pd.DataFrame:
    raw = read_tss_workbook(xlsx_path)
    cleaned = clean_tss_dataframe(raw, allow_n=allow_n)
    usable = cleaned.loc[cleaned["qc_pass"]].reset_index(drop=True)
    if usable.empty:
        raise ValueError("No TSS/PAS rows passed QC")
    return assign_grouped_splits(
        usable,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )


def dna_one_hot(sequence: str) -> np.ndarray:
    mapping = {
        "A": (1.0, 0.0, 0.0, 0.0),
        "C": (0.0, 1.0, 0.0, 0.0),
        "G": (0.0, 0.0, 1.0, 0.0),
        "T": (0.0, 0.0, 0.0, 1.0),
        "N": (0.25, 0.25, 0.25, 0.25),
    }
    return np.asarray([mapping[base] for base in sequence], dtype=np.float32).T


def architecture_crop(row: pd.Series, target_position: int) -> tuple[str, int]:
    if not 0 <= target_position < MODEL_N_POSITIONS:
        raise ValueError(f"target_position must be in [0, {MODEL_N_POSITIONS - 1}]")
    crop_start = int(row["m35_start"]) - (MODEL_M35_OFFSET + target_position)
    crop_end = crop_start + MODEL_INPUT_LENGTH
    sequence = str(row["full_sequence"])
    if crop_start < 0 or crop_end > len(sequence):
        raise ValueError(
            f"Cannot make 80-bp crop for {row['source_sheet']} row {row['source_row']}: "
            f"start={crop_start}, end={crop_end}, sequence_length={len(sequence)}"
        )
    crop = sequence[crop_start:crop_end]
    target_index = int(row["design_channel"]) * MODEL_N_POSITIONS + target_position
    return crop, target_index


class TSSArchitectureDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        randomize_position: bool,
        seed: int = SEED,
        fixed_position: int = MODEL_N_POSITIONS // 2,
    ):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.randomize_position = bool(randomize_position)
        self.seed = int(seed)
        self.fixed_position = int(fixed_position)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _target_position(self, index: int) -> int:
        if not self.randomize_position:
            return self.fixed_position
        # Index/epoch-derived RNG keeps runs reproducible with DataLoader(shuffle=True).
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch + index)
        return int(rng.integers(0, MODEL_N_POSITIONS))

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        target_position = self._target_position(index)
        crop, target_index = architecture_crop(row, target_position)
        return (
            torch.from_numpy(dna_one_hot(crop)),
            torch.tensor(target_index, dtype=torch.long),
            torch.tensor(int(row["design_channel"]), dtype=torch.long),
        )


def qc_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    records = [
        {"metric": "total_rows", "value": int(len(cleaned))},
        {"metric": "qc_pass_rows", "value": int(cleaned["qc_pass"].sum())},
        {"metric": "qc_fail_rows", "value": int((~cleaned["qc_pass"]).sum())},
    ]
    records.extend(
        {"metric": f"spacer_length_{length}", "value": int(count)}
        for length, count in cleaned["spacer_length"].value_counts().sort_index().items()
    )
    return pd.DataFrame(records)


def save_processed_dataset(
    xlsx_path: Path | str = DEFAULT_XLSX,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = SEED,
    allow_n: bool = True,
) -> dict[str, Path]:
    xlsx_path = Path(xlsx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_tss_workbook(xlsx_path)
    cleaned = clean_tss_dataframe(raw, allow_n=allow_n)
    usable = assign_grouped_splits(
        cleaned.loc[cleaned["qc_pass"]].reset_index(drop=True),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    dataset_path = output_dir / "tss_pas_processed.pkl"
    qc_path = output_dir / "tss_pas_qc_summary.csv"
    sheet_path = output_dir / "tss_pas_sheet_summary.csv"
    split_path = output_dir / "tss_pas_split_summary.csv"
    groups_path = output_dir / "tss_pas_split_groups.csv"
    failure_summary_path = output_dir / "tss_pas_qc_failure_summary.csv"
    failures_path = output_dir / "tss_pas_qc_failures.csv"
    config_path = output_dir / "tss_pas_dataset_config.json"

    usable.to_pickle(dataset_path)
    qc_summary(cleaned).to_csv(qc_path, index=False)
    cleaned.groupby("source_sheet", as_index=False).agg(
        rows=("source_row", "size"),
        qc_pass_rows=("qc_pass", "sum"),
    ).to_csv(sheet_path, index=False)
    usable.groupby(["split", "spacer_length"], as_index=False, observed=True).size().rename(
        columns={"size": "rows"}
    ).to_csv(split_path, index=False)
    usable[["species_group", "split"]].drop_duplicates().sort_values(
        ["split", "species_group"]
    ).to_csv(groups_path, index=False)
    failures = cleaned.loc[~cleaned["qc_pass"]].copy()
    failures["qc_reason"].value_counts().rename_axis("qc_reason").reset_index(
        name="rows"
    ).to_csv(failure_summary_path, index=False)
    failures[
        ["source_sheet", "source_row", "species_group", *SEQUENCE_COLUMNS, "qc_reason"]
    ].to_csv(failures_path, index=False)
    config = TSSDataConfig(
        xlsx_path=str(xlsx_path.resolve()),
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        allow_n=allow_n,
    )
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return {
        "dataset": dataset_path,
        "qc_summary": qc_path,
        "sheet_summary": sheet_path,
        "split_summary": split_path,
        "split_groups": groups_path,
        "qc_failure_summary": failure_summary_path,
        "qc_failures": failures_path,
        "config": config_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TSS/PAS architecture data for CorePromoter.")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reject-n", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = save_processed_dataset(
        xlsx_path=args.xlsx,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        allow_n=not args.reject_n,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
