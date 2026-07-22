from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import recursive_corepromoter_design as legacy
from tss_pas_dataset import (
    DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_DIR,
    DEFAULT_XLSX,
    MODEL_N_POSITIONS,
    TSSArchitectureDataset,
    save_processed_dataset,
)


DEFAULT_BASELINE = legacy.WEIGHTS_DIR / "weights_CorePromoter_clean.pt"
DEFAULT_DATASET = DEFAULT_DATASET_DIR / "tss_pas_processed.pkl"
DEFAULT_FINAL_CHECKPOINT = legacy.WEIGHTS_DIR / "weights_CorePromoter_tss_pas.pt"
DEFAULT_ARCH_CHECKPOINT = legacy.WEIGHTS_DIR / "weights_CorePromoter_tss_arch.pt"
DEFAULT_RUN_ROOT = legacy.PROJECT_ROOT / "outputs" / "corepromoter_tss_pas"
HEAD_PARAMETER_NAMES = ("conds_bias", "param_max", "param_min", "param_e0")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = legacy.SEED
    batch_size: int = 512
    arch_epochs: int = 20
    head_epochs: int = 15
    joint_epochs: int = 15
    arch_lr: float = 1e-4
    head_lr: float = 1e-3
    joint_backbone_lr: float = 1e-5
    joint_head_lr: float = 1e-4
    weight_decay: float = 1e-5
    hard_negative_margin: float = 0.3
    hard_negative_weight: float = 0.2
    shuffled_negative_weight: float = 0.1
    joint_architecture_weight: float = 0.2
    expression_validation_fraction: float = 0.2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_core_checkpoint(path: Path, device: torch.device) -> legacy.CorePromoterModel:
    checkpoint = safe_torch_load(path, device)
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        seq_length = int(checkpoint.get("seq_length", 80))
        num_conds = int(checkpoint.get("num_conds", len(legacy.LIBRARY_ORDER)))
    else:
        state = checkpoint
        seq_length = 80
        cond_bias = state.get("conds_bias")
        num_conds = int(cond_bias.numel()) if cond_bias is not None else len(legacy.LIBRARY_ORDER)
    model = legacy.CorePromoterModel(seq_length=seq_length, num_conds=num_conds).to(device)
    model.load_state_dict(state, strict=True)
    return model


def save_core_checkpoint(
    model: legacy.CorePromoterModel,
    path: Path,
    *,
    stage: str,
    source_checkpoint: Path,
    config: TrainingConfig,
    condition_bias_centered: bool,
    extra_metadata: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "seq_length": int(model.seq_length),
        "num_conds": int(model.conds_bias.numel()),
        "seed": config.seed,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": file_sha256(source_checkpoint),
        "conv2_weight_frozen": True,
        "condition_bias_centered": bool(condition_bias_centered),
        "training_config": asdict(config),
    }
    if extra_metadata:
        checkpoint.update(extra_metadata)
    torch.save(checkpoint, path)


def set_trainable(model: legacy.CorePromoterModel, names: Iterable[str]) -> list[nn.Parameter]:
    selected = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in selected
    # The sparse conv2 mask is a scanner invariant and must never be optimized.
    model.conv2.weight.requires_grad = False
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def center_condition_bias(model: legacy.CorePromoterModel) -> None:
    with torch.no_grad():
        model.conds_bias.sub_(model.conds_bias.mean())


def channel_weights(dataframe: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = dataframe["design_channel"].value_counts().reindex(range(3), fill_value=0).to_numpy(float)
    if np.any(counts <= 0):
        raise ValueError(f"Every spacer channel must occur in training data; counts={counts.tolist()}")
    values = counts.sum() / (len(counts) * counts)
    values /= values.mean()
    return torch.tensor(values, dtype=torch.float32, device=device)


def composition_shuffle(x: torch.Tensor) -> torch.Tensor:
    batch, channels, length = x.shape
    permutations = torch.argsort(torch.rand(batch, length, device=x.device), dim=1)
    return x.gather(2, permutations.unsqueeze(1).expand(batch, channels, length))


def architecture_loss(
    model: legacy.CorePromoterModel,
    x: torch.Tensor,
    target_index: torch.Tensor,
    target_channel: torch.Tensor,
    *,
    spacer_channel_weights: torch.Tensor,
    margin: float,
    hard_negative_weight: float,
    shuffled_negative_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = model.architecture_logits(x).flatten(1)
    if logits.shape[1] != 3 * MODEL_N_POSITIONS:
        raise ValueError(f"Expected 27 candidate scores for 80-bp crops; received {logits.shape}")
    ce_each = nn.functional.cross_entropy(logits, target_index, reduction="none")
    ce = (ce_each * spacer_channel_weights[target_channel]).mean()

    target_score = logits.gather(1, target_index.unsqueeze(1)).squeeze(1)
    non_target = logits.clone()
    non_target.scatter_(1, target_index.unsqueeze(1), float("-inf"))
    strongest_non_target = non_target.max(dim=1).values
    hard_margin = nn.functional.relu(margin - target_score + strongest_non_target).mean()

    shuffled = composition_shuffle(x)
    shuffled_max = model.architecture_logits(shuffled).flatten(1).max(dim=1).values
    shuffled_margin = nn.functional.relu(margin - target_score + shuffled_max).mean()
    total = ce + hard_negative_weight * hard_margin + shuffled_negative_weight * shuffled_margin
    return total, {
        "cross_entropy": ce.detach(),
        "hard_margin": hard_margin.detach(),
        "shuffled_margin": shuffled_margin.detach(),
    }


def architecture_metrics(
    model: legacy.CorePromoterModel,
    dataset: TSSArchitectureDataset,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predicted, targets, margins = [], [], []
    model.eval()
    with torch.no_grad():
        for x, target_index, _ in loader:
            x = x.to(device)
            target_index = target_index.to(device)
            logits = model.architecture_logits(x).flatten(1)
            pred = logits.argmax(dim=1)
            target_score = logits.gather(1, target_index.unsqueeze(1)).squeeze(1)
            non_target = logits.clone()
            non_target.scatter_(1, target_index.unsqueeze(1), float("-inf"))
            margin = target_score - non_target.max(dim=1).values
            predicted.append(pred.cpu().numpy())
            targets.append(target_index.cpu().numpy())
            margins.append(margin.cpu().numpy())
    pred = np.concatenate(predicted)
    target = np.concatenate(targets)
    margin = np.concatenate(margins)
    pred_channel, pred_position = pred // MODEL_N_POSITIONS, pred % MODEL_N_POSITIONS
    target_channel, target_position = target // MODEL_N_POSITIONS, target % MODEL_N_POSITIONS
    pred_m10 = pred_position + np.array([legacy.SPACER_BY_CHANNEL[int(ch)] for ch in pred_channel])
    target_m10 = target_position + np.array([legacy.SPACER_BY_CHANNEL[int(ch)] for ch in target_channel])
    return {
        "n": float(len(target)),
        "top1_architecture_accuracy": float(np.mean(pred == target)),
        "m35_position_accuracy": float(np.mean(pred_position == target_position)),
        "m10_position_accuracy": float(np.mean(pred_m10 == target_m10)),
        "spacer_accuracy": float(np.mean(pred_channel == target_channel)),
        "mean_target_margin": float(np.mean(margin)),
        "median_target_margin": float(np.median(margin)),
        "positive_margin_rate": float(np.mean(margin > 0)),
    }


def run_architecture_pretraining(
    model: legacy.CorePromoterModel,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    device: torch.device,
    config: TrainingConfig,
) -> list[dict]:
    train_dataset = TSSArchitectureDataset(train_data, randomize_position=True, seed=config.seed)
    validation_dataset = TSSArchitectureDataset(validation_data, randomize_position=False, seed=config.seed)
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    parameters = set_trainable(model, {"conv1.weight", "conv2.bias"})
    optimizer = optim.Adam(parameters, lr=config.arch_lr, weight_decay=config.weight_decay)
    weights = channel_weights(train_data, device)
    history: list[dict] = []
    for epoch in range(config.arch_epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        total = 0.0
        for x, target_index, target_channel in loader:
            x = x.to(device)
            target_index = target_index.to(device)
            target_channel = target_channel.to(device)
            optimizer.zero_grad()
            loss, _ = architecture_loss(
                model,
                x,
                target_index,
                target_channel,
                spacer_channel_weights=weights,
                margin=config.hard_negative_margin,
                hard_negative_weight=config.hard_negative_weight,
                shuffled_negative_weight=config.shuffled_negative_weight,
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(x)
        metrics = architecture_metrics(
            model, validation_dataset, device, batch_size=config.batch_size
        )
        record = {
            "stage": "architecture",
            "epoch": epoch + 1,
            "train_loss": total / len(train_dataset),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(record)
        print(
            f"Architecture [{epoch + 1:02d}/{config.arch_epochs}] "
            f"loss={record['train_loss']:.4f} "
            f"top1={metrics['top1_architecture_accuracy']:.4f} "
            f"margin={metrics['mean_target_margin']:.4f}"
        )
    return history


def expression_arrays(dataframe: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    categorical = pd.Categorical(dataframe["Library"], categories=legacy.LIBRARY_ORDER)
    conditions = pd.get_dummies(categorical).to_numpy(dtype=np.float32)
    sequences = np.asarray(
        [legacy.dna_one_hot(sequence) for sequence in dataframe["Sequence"]], dtype=np.float32
    )
    targets = (dataframe["LogGFP"].to_numpy(np.float32) * np.log(10)).reshape(-1, 1)
    return sequences, conditions, targets


def expression_loaders(
    dataframe: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    indices = np.arange(len(dataframe))
    train_idx, validation_idx = train_test_split(
        indices,
        test_size=config.expression_validation_fraction,
        random_state=config.seed,
        stratify=dataframe["Library"],
    )
    train_df = dataframe.iloc[train_idx].reset_index(drop=True)
    validation_df = dataframe.iloc[validation_idx].reset_index(drop=True)

    def loader_for(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        x, conditions, targets = expression_arrays(frame)
        dataset = TensorDataset(
            torch.from_numpy(x), torch.from_numpy(conditions), torch.from_numpy(targets)
        )
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)

    return loader_for(train_df, True), loader_for(validation_df, False), train_df, validation_df


def expression_metrics(
    model: legacy.CorePromoterModel,
    dataframe: pd.DataFrame,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, float]:
    x, conditions, targets = expression_arrays(dataframe)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(conditions), torch.from_numpy(targets))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions, observed = [], []
    model.eval()
    with torch.no_grad():
        for xb, cb, yb in loader:
            pred = model(xb.to(device), cb.to(device)) / np.log(10)
            predictions.append(pred.cpu().numpy().ravel())
            observed.append((yb.numpy().ravel() / np.log(10)))
    pred = np.concatenate(predictions)
    obs = np.concatenate(observed)
    residual = pred - obs
    ss_res = float(np.sum(residual**2))
    ss_total = float(np.sum((obs - obs.mean()) ** 2))
    pearson = (
        float(np.corrcoef(obs, pred)[0, 1])
        if np.std(obs) > 0 and np.std(pred) > 0
        else float("nan")
    )
    spearman = (
        float(spearmanr(obs, pred).statistic)
        if np.std(obs) > 0 and np.std(pred) > 0
        else float("nan")
    )
    return {
        "n": float(len(obs)),
        "mse_log10": float(np.mean(residual**2)),
        "rmse_log10": float(np.sqrt(np.mean(residual**2))),
        "mae_log10": float(np.mean(np.abs(residual))),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "r2": float(1 - ss_res / ss_total) if ss_total > 0 else float("nan"),
        "residual_mean": float(residual.mean()),
    }


def run_expression_head_recalibration(
    model: legacy.CorePromoterModel,
    train_loader: DataLoader,
    validation_data: pd.DataFrame,
    device: torch.device,
    config: TrainingConfig,
) -> list[dict]:
    parameters = set_trainable(model, HEAD_PARAMETER_NAMES)
    optimizer = optim.Adam(parameters, lr=config.head_lr, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    history: list[dict] = []
    for epoch in range(config.head_epochs):
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
        metrics = expression_metrics(
            model, validation_data, device, batch_size=config.batch_size
        )
        record = {
            "stage": "expression_head",
            "epoch": epoch + 1,
            "train_loss_loge": total / rows,
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(record)
        print(
            f"Expression head [{epoch + 1:02d}/{config.head_epochs}] "
            f"train_loge={record['train_loss_loge']:.4f} "
            f"validation_mse_log10={metrics['mse_log10']:.4f}"
        )
    return history


def run_joint_finetuning(
    model: legacy.CorePromoterModel,
    expression_train_loader: DataLoader,
    expression_validation: pd.DataFrame,
    architecture_train: pd.DataFrame,
    architecture_validation: pd.DataFrame,
    device: torch.device,
    config: TrainingConfig,
) -> list[dict]:
    architecture_dataset = TSSArchitectureDataset(
        architecture_train, randomize_position=True, seed=config.seed + 100
    )
    architecture_loader = DataLoader(
        architecture_dataset, batch_size=config.batch_size, shuffle=True
    )
    validation_dataset = TSSArchitectureDataset(
        architecture_validation, randomize_position=False, seed=config.seed
    )
    set_trainable(model, {"conv1.weight", "conv2.bias", *HEAD_PARAMETER_NAMES})
    optimizer = optim.Adam(
        [
            {"params": [model.conv1.weight, model.conv2.bias], "lr": config.joint_backbone_lr},
            {
                "params": [model.conds_bias, model.param_max, model.param_min, model.param_e0],
                "lr": config.joint_head_lr,
            },
        ],
        weight_decay=config.weight_decay,
    )
    criterion = nn.MSELoss()
    weights = channel_weights(architecture_train, device)
    history: list[dict] = []
    for epoch in range(config.joint_epochs):
        architecture_dataset.set_epoch(epoch)
        architecture_iter = iter(architecture_loader)
        model.train()
        total = 0.0
        rows = 0
        for x_expr, conditions, target_expr in expression_train_loader:
            try:
                x_arch, target_index, target_channel = next(architecture_iter)
            except StopIteration:
                architecture_iter = iter(architecture_loader)
                x_arch, target_index, target_channel = next(architecture_iter)
            x_expr = x_expr.to(device)
            conditions = conditions.to(device)
            target_expr = target_expr.to(device)
            x_arch = x_arch.to(device)
            target_index = target_index.to(device)
            target_channel = target_channel.to(device)
            optimizer.zero_grad()
            expression_loss = criterion(model(x_expr, conditions), target_expr)
            arch_loss, _ = architecture_loss(
                model,
                x_arch,
                target_index,
                target_channel,
                spacer_channel_weights=weights,
                margin=config.hard_negative_margin,
                hard_negative_weight=config.hard_negative_weight,
                shuffled_negative_weight=config.shuffled_negative_weight,
            )
            loss = expression_loss + config.joint_architecture_weight * arch_loss
            loss.backward()
            optimizer.step()
            center_condition_bias(model)
            total += float(loss.detach()) * len(x_expr)
            rows += len(x_expr)
        expr_metrics = expression_metrics(
            model, expression_validation, device, batch_size=config.batch_size
        )
        arch_metrics = architecture_metrics(
            model, validation_dataset, device, batch_size=config.batch_size
        )
        record = {
            "stage": "joint",
            "epoch": epoch + 1,
            "train_loss": total / rows,
            **{f"expression_validation_{key}": value for key, value in expr_metrics.items()},
            **{f"architecture_validation_{key}": value for key, value in arch_metrics.items()},
        }
        history.append(record)
        print(
            f"Joint [{epoch + 1:02d}/{config.joint_epochs}] loss={record['train_loss']:.4f} "
            f"expr_mse={expr_metrics['mse_log10']:.4f} "
            f"arch_top1={arch_metrics['top1_architecture_accuracy']:.4f}"
        )
    return history


def metrics_records(prefix: str, metrics: dict[str, float]) -> list[dict]:
    return [{"evaluation": prefix, "metric": key, "value": value} for key, value in metrics.items()]


def train_production_model(
    *,
    dataset_path: Path,
    baseline_checkpoint: Path,
    arch_checkpoint: Path,
    final_checkpoint: Path,
    run_dir: Path,
    device: torch.device,
    config: TrainingConfig,
    max_tss_rows: int | None = None,
) -> dict[str, Path]:
    legacy.set_seed(config.seed)
    tss_data = pd.read_pickle(dataset_path)
    if max_tss_rows is not None:
        tss_data = pd.concat(
            [
                frame.sample(min(len(frame), max_tss_rows), random_state=config.seed)
                for _, frame in tss_data.groupby("split", sort=False)
            ],
            ignore_index=True,
        )
    train_tss = tss_data.loc[tss_data["split"] == "train"].reset_index(drop=True)
    validation_tss = tss_data.loc[tss_data["split"] == "validation"].reset_index(drop=True)
    test_tss = tss_data.loc[tss_data["split"] == "test"].reset_index(drop=True)

    model = load_core_checkpoint(baseline_checkpoint, device)
    baseline_model = copy.deepcopy(model).eval()
    fixed_validation = TSSArchitectureDataset(validation_tss, randomize_position=False)
    fixed_test = TSSArchitectureDataset(test_tss, randomize_position=False)
    metrics: list[dict] = []
    metrics += metrics_records(
        "baseline_architecture_validation",
        architecture_metrics(baseline_model, fixed_validation, device, batch_size=config.batch_size),
    )
    metrics += metrics_records(
        "baseline_architecture_test",
        architecture_metrics(baseline_model, fixed_test, device, batch_size=config.batch_size),
    )

    history = run_architecture_pretraining(model, train_tss, validation_tss, device, config)
    save_core_checkpoint(
        model,
        arch_checkpoint,
        stage="tss_pas_architecture_pretraining",
        source_checkpoint=baseline_checkpoint,
        config=config,
        condition_bias_centered=False,
        extra_metadata={"dataset_path": str(dataset_path.resolve())},
    )

    expression_data = legacy.build_core_training_dataframe()
    expression_train_loader, _, _, expression_validation = expression_loaders(expression_data, config)
    metrics += metrics_records(
        "baseline_expression_validation",
        expression_metrics(
            baseline_model, expression_validation, device, batch_size=config.batch_size
        ),
    )
    history += run_expression_head_recalibration(
        model, expression_train_loader, expression_validation, device, config
    )
    history += run_joint_finetuning(
        model,
        expression_train_loader,
        expression_validation,
        train_tss,
        validation_tss,
        device,
        config,
    )
    center_condition_bias(model)

    final_arch_validation = architecture_metrics(
        model, fixed_validation, device, batch_size=config.batch_size
    )
    final_arch_test = architecture_metrics(model, fixed_test, device, batch_size=config.batch_size)
    final_expression_validation = expression_metrics(
        model, expression_validation, device, batch_size=config.batch_size
    )
    metrics += metrics_records("tss_pas_architecture_validation", final_arch_validation)
    metrics += metrics_records("tss_pas_architecture_test", final_arch_test)
    metrics += metrics_records("tss_pas_expression_validation", final_expression_validation)

    save_core_checkpoint(
        model,
        final_checkpoint,
        stage="tss_pas_expression_joint_finetuning",
        source_checkpoint=baseline_checkpoint,
        config=config,
        condition_bias_centered=True,
        extra_metadata={
            "dataset_path": str(dataset_path.resolve()),
            "architecture_validation": final_arch_validation,
            "architecture_test": final_arch_test,
            "expression_validation": final_expression_validation,
        },
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training_history.csv"
    metrics_path = run_dir / "model_metrics.csv"
    config_path = run_dir / "training_config.json"
    pd.DataFrame(history).to_csv(history_path, index=False)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                **asdict(config),
                "dataset_path": str(dataset_path.resolve()),
                "baseline_checkpoint": str(baseline_checkpoint.resolve()),
                "arch_checkpoint": str(arch_checkpoint.resolve()),
                "final_checkpoint": str(final_checkpoint.resolve()),
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "arch_checkpoint": arch_checkpoint,
        "final_checkpoint": final_checkpoint,
        "history": history_path,
        "metrics": metrics_path,
        "config": config_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSS/PAS pretrain and expression-calibrate CorePromoter.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--arch-checkpoint", type=Path, default=DEFAULT_ARCH_CHECKPOINT)
    parser.add_argument("--final-checkpoint", type=Path, default=DEFAULT_FINAL_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--arch-epochs", type=int, default=TrainingConfig.arch_epochs)
    parser.add_argument("--head-epochs", type=int, default=TrainingConfig.head_epochs)
    parser.add_argument("--joint-epochs", type=int, default=TrainingConfig.joint_epochs)
    parser.add_argument("--max-tss-rows", type=int)
    parser.add_argument("--prepare-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_data or not args.dataset.exists():
        outputs = save_processed_dataset(args.xlsx, args.dataset.parent)
        dataset_path = outputs["dataset"]
    else:
        dataset_path = args.dataset
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    config = TrainingConfig(
        batch_size=args.batch_size,
        arch_epochs=args.arch_epochs,
        head_epochs=args.head_epochs,
        joint_epochs=args.joint_epochs,
    )
    run_dir = args.run_dir or DEFAULT_RUN_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = train_production_model(
        dataset_path=dataset_path,
        baseline_checkpoint=args.baseline_checkpoint,
        arch_checkpoint=args.arch_checkpoint,
        final_checkpoint=args.final_checkpoint,
        run_dir=run_dir,
        device=device,
        config=config,
        max_tss_rows=args.max_tss_rows,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
