from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import recursive_corepromoter_design as legacy
from train_corepromoter_tss_pas import (
    DEFAULT_BASELINE,
    DEFAULT_DATASET,
    DEFAULT_FINAL_CHECKPOINT,
    HEAD_PARAMETER_NAMES,
    TrainingConfig,
    architecture_metrics,
    center_condition_bias,
    expression_loaders,
    expression_metrics,
    load_core_checkpoint,
    run_architecture_pretraining,
    run_expression_head_recalibration,
    run_joint_finetuning,
    set_trainable,
)
from tss_pas_dataset import TSSArchitectureDataset


DEFAULT_OUTPUT_ROOT = legacy.PROJECT_ROOT / "outputs" / "corepromoter_tss_pas_evaluation"


def checkpoint_comparison(
    baseline_checkpoint: Path,
    tss_checkpoint: Path,
    tss_data: pd.DataFrame,
    expression_data: pd.DataFrame,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    test_tss = tss_data.loc[tss_data["split"] == "test"].reset_index(drop=True)
    test_dataset = TSSArchitectureDataset(test_tss, randomize_position=False)
    records: list[dict] = []
    for model_name, path in (("baseline", baseline_checkpoint), ("tss_pas", tss_checkpoint)):
        model = load_core_checkpoint(path, device).eval()
        arch = architecture_metrics(model, test_dataset, device, batch_size=batch_size)
        records.extend(
            {
                "model": model_name,
                "evaluation": "species_group_held_out_architecture_test",
                "group": "all_test_species",
                "metric": metric,
                "value": value,
            }
            for metric, value in arch.items()
        )
        for library in legacy.LIBRARY_ORDER:
            library_data = expression_data.loc[expression_data["Library"] == library].reset_index(drop=True)
            metrics = expression_metrics(model, library_data, device, batch_size=batch_size)
            records.extend(
                {
                    "model": model_name,
                    "evaluation": "checkpoint_per_library_diagnostic",
                    "group": library,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in metrics.items()
            )
    return pd.DataFrame(records)


def train_expression_baseline(
    model: legacy.CorePromoterModel,
    train_loader,
    validation_data: pd.DataFrame,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> list[dict]:
    parameters = set_trainable(model, {"conv1.weight", "conv2.bias", *HEAD_PARAMETER_NAMES})
    optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        rows = 0
        for x, conditions, target in train_loader:
            x, conditions, target = x.to(device), conditions.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, conditions), target)
            loss.backward()
            optimizer.step()
            center_condition_bias(model)
            total += float(loss.detach()) * len(x)
            rows += len(x)
        metrics = expression_metrics(model, validation_data, device, batch_size=batch_size)
        history.append(
            {
                "stage": "lolo_expression_baseline",
                "epoch": epoch + 1,
                "train_loss_loge": total / rows,
                "validation_mse_log10": metrics["mse_log10"],
            }
        )
    return history


def leave_one_library_out_comparison(
    expression_data: pd.DataFrame,
    tss_data: pd.DataFrame,
    device: torch.device,
    config: TrainingConfig,
    *,
    baseline_epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_tss = tss_data.loc[tss_data["split"] == "train"].reset_index(drop=True)
    validation_tss = tss_data.loc[tss_data["split"] == "validation"].reset_index(drop=True)
    records: list[dict] = []
    history_records: list[dict] = []
    for fold_index, held_out in enumerate(legacy.LIBRARY_ORDER):
        print(f"LOLO fold {fold_index + 1}/{len(legacy.LIBRARY_ORDER)}: hold out {held_out}")
        legacy.set_seed(config.seed + fold_index)
        fold_train = expression_data.loc[expression_data["Library"] != held_out].reset_index(drop=True)
        fold_test = expression_data.loc[expression_data["Library"] == held_out].reset_index(drop=True)
        train_loader, _, _, validation_data = expression_loaders(fold_train, config)

        baseline_model = legacy.CorePromoterModel(
            seq_length=80, num_conds=len(legacy.LIBRARY_ORDER)
        ).to(device)
        baseline_history = train_expression_baseline(
            baseline_model,
            train_loader,
            validation_data,
            device,
            epochs=baseline_epochs,
            lr=1e-3,
            weight_decay=config.weight_decay,
            batch_size=config.batch_size,
        )
        for row in baseline_history:
            history_records.append({"held_out_library": held_out, "model": "baseline", **row})
        baseline_test = expression_metrics(
            baseline_model, fold_test, device, batch_size=config.batch_size
        )
        records.extend(
            {
                "held_out_library": held_out,
                "model": "baseline",
                "metric": metric,
                "value": value,
            }
            for metric, value in baseline_test.items()
        )

        tss_model = copy.deepcopy(baseline_model)
        fold_history = run_architecture_pretraining(
            tss_model, train_tss, validation_tss, device, config
        )
        fold_history += run_expression_head_recalibration(
            tss_model, train_loader, validation_data, device, config
        )
        fold_history += run_joint_finetuning(
            tss_model,
            train_loader,
            validation_data,
            train_tss,
            validation_tss,
            device,
            config,
        )
        for row in fold_history:
            history_records.append({"held_out_library": held_out, "model": "tss_pas", **row})
        tss_test = expression_metrics(tss_model, fold_test, device, batch_size=config.batch_size)
        records.extend(
            {
                "held_out_library": held_out,
                "model": "tss_pas",
                "metric": metric,
                "value": value,
            }
            for metric, value in tss_test.items()
        )
    return pd.DataFrame(records), pd.DataFrame(history_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and TSS/PAS CorePromoter models.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--tss-checkpoint", type=Path, default=DEFAULT_FINAL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--run-lolo", action="store_true")
    parser.add_argument("--lolo-baseline-epochs", type=int, default=50)
    parser.add_argument("--lolo-arch-epochs", type=int, default=20)
    parser.add_argument("--lolo-head-epochs", type=int, default=15)
    parser.add_argument("--lolo-joint-epochs", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    tss_data = pd.read_pickle(args.dataset)
    expression_data = legacy.build_core_training_dataframe()

    checkpoint_metrics = checkpoint_comparison(
        args.baseline_checkpoint,
        args.tss_checkpoint,
        tss_data,
        expression_data,
        device,
        args.batch_size,
    )
    checkpoint_path = output_dir / "checkpoint_comparison_metrics.csv"
    checkpoint_metrics.to_csv(checkpoint_path, index=False)
    print(f"checkpoint_comparison: {checkpoint_path}")

    if args.run_lolo:
        config = TrainingConfig(
            batch_size=args.batch_size,
            arch_epochs=args.lolo_arch_epochs,
            head_epochs=args.lolo_head_epochs,
            joint_epochs=args.lolo_joint_epochs,
        )
        lolo_metrics, lolo_history = leave_one_library_out_comparison(
            expression_data,
            tss_data,
            device,
            config,
            baseline_epochs=args.lolo_baseline_epochs,
        )
        lolo_path = output_dir / "leave_one_library_out_metrics.csv"
        history_path = output_dir / "leave_one_library_out_history.csv"
        lolo_metrics.to_csv(lolo_path, index=False)
        lolo_history.to_csv(history_path, index=False)
        print(f"leave_one_library_out_metrics: {lolo_path}")
        print(f"leave_one_library_out_history: {history_path}")


if __name__ == "__main__":
    main()
