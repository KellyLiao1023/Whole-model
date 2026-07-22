from __future__ import annotations

import itertools
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


SEED = 777
LIBRARY_ORDER = ["PL17", "SL16", "SL17", "SL18", "DL", "UL", "ITS"]
ORDER = ["UP", "m35", "spacer", "m10", "DIS", "ITS"]
BG5 = "AGGGAAGAGACC"
BG3 = "GTCGACTCTAGA"
LINKER = "CAC"
M35_LEN = 6
M10_LEN = 6
SPACERS = [16, 17, 18]
SPACER_BY_CHANNEL = {ch: sp for ch, sp in enumerate(SPACERS)}
CHANNEL_BY_SPACER = {sp: ch for ch, sp in SPACER_BY_CHANNEL.items()}


@dataclass(frozen=True)
class SlotKey:
    element: str
    slot_id: str


@dataclass
class RecursiveDesignResult:
    success: bool
    best_summary: dict
    out_dir: Path
    history: pd.DataFrame
    final_elements: pd.DataFrame
    final_badness: pd.DataFrame
    best_scan: pd.DataFrame


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for base in [start, *start.parents]:
        if (base / "MS2_Data_PyTorch" / "tables").exists():
            return base
        if base.name == "MS2_Data_PyTorch" and (base / "tables").exists():
            return base.parent
    raise FileNotFoundError("Could not find project root containing MS2_Data_PyTorch/tables")


PROJECT_ROOT = find_project_root()
MS2_DIR = PROJECT_ROOT / "MS2_Data_PyTorch"
SCRIPT_DIR = MS2_DIR / "scripts"
TABLE_DIR = MS2_DIR / "tables"
WEIGHTS_DIR = MS2_DIR / "weights"
DEFAULT_PARENT_OUT = PROJECT_ROOT / "outputs" / "019ec8ae-c0ac-7190-868c-5c4ba74a5396"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from util import Motif2Seqs, getPFM  # noqa: E402
from BPM.BPM import predict as bpm_predict  # noqa: E402


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dna_one_hot(seq: str, flatten: bool = False) -> np.ndarray:
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "-": [-1, -1, -1, -1],
        "N": [0.25, 0.25, 0.25, 0.25],
    }
    one_hot = np.array([mapping.get(base, [0, 0, 0, 0]) for base in str(seq).upper()]).T
    return one_hot.flatten() if flatten else one_hot


def stratified_sample(df: pd.DataFrame, col: str, counts: list[int], seed: int = SEED) -> pd.DataFrame:
    df = df.copy()
    rng_bins = pd.cut(df[col], bins=len(counts))
    parts = []
    for grp, cnt in zip(sorted(rng_bins.dropna().unique()), counts):
        sub = df[rng_bins == grp]
        if len(sub) > 0:
            parts.append(sub.sample(cnt, replace=True, random_state=seed))
    return pd.concat(parts)


def torch_load_weights(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def PpurR(seq: str) -> str:
    return "CACAGCAGCAGTCAGGTACTCCAGTCCAC" + seq[:6] + "TTTTCGTCAAGATCGGC" + seq[-6:] + "TCCACGCTTACTAAATCTATGT"


def SL16(seq: str) -> str:
    return "ACAGCAGCAGTCAGGTACTCCAGTAAATGCTTGACT" + seq + "TATTATGCACACCCCTAAATCTATGTAG"


def SL17(seq: str) -> str:
    return "CACAGCAGCAGTCAGGTACTCCAGTCCACTTGACC" + seq + "TAACCTTCCACGCTTACTAAATCTATGT"


def SL18(seq: str) -> str:
    return "TCTTCACACAGCAGCTCAGGTACTCAGGCTTTACA" + seq + "GATAATGTGTGGAATTAAATCTATGTA"


def DL(seq: str) -> str:
    return "AGCAGCAGTCAGGTACTCCAGTAAATGCTTGCCCGCCGCGTGATTCGTGTTATAAC" + seq + "CAAAATCTATGTAGCT"


def UL(seq: str) -> str:
    return "CAGAAAAAG" + seq + "GGCTTGCGGCTTTTGCCGCTTTTTTTTACCCTGCACACCCCT----------"


def ITS_context(seq: str) -> str:
    return "AGCAGCGTTAAATTCACGCCCTTCTCTTGAGACATTTCTTTTGCACTGGTAAACTAAATC" + seq + "GTCCCAGGCT"


def build_core_training_dataframe() -> pd.DataFrame:
    df_pl = stratified_sample(pd.read_pickle(TABLE_DIR / "PL.pkl"), "LogGFP", [2000, 2000, 2000, 1000, 1000, 1000, 500, 500])[["rep1", "rep2", "LogGFP"]]
    df_pl["Sequence"] = df_pl.index.map(PpurR)
    df_pl["Library"] = "PL17"

    df_sl16 = stratified_sample(pd.read_pickle(TABLE_DIR / "SL16.pkl"), "LogGFP", [300] * 10)[["rep1", "rep2", "LogGFP"]]
    df_sl16["Sequence"] = df_sl16.index.map(SL16)
    df_sl16["Library"] = "SL16"

    df_sl17 = stratified_sample(pd.read_pickle(TABLE_DIR / "SL17.pkl"), "LogGFP", [400] * 10)[["rep1", "rep2", "LogGFP"]]
    df_sl17["Sequence"] = df_sl17.index.map(SL17)
    df_sl17["Library"] = "SL17"

    df_sl18 = stratified_sample(pd.read_pickle(TABLE_DIR / "SL18.pkl"), "LogGFP", [300] * 10)[["rep1", "rep2", "LogGFP"]]
    df_sl18["Sequence"] = df_sl18.index.map(SL18)
    df_sl18["Library"] = "SL18"

    df_dl = stratified_sample(pd.read_pickle(TABLE_DIR / "DL.pkl"), "LogGFP", [200] * 10)[["rep1", "rep2", "LogGFP"]]
    df_dl["Sequence"] = df_dl.index.map(DL)
    df_dl["Library"] = "DL"

    df_ul = pd.read_pickle(TABLE_DIR / "UL.pkl")
    df_ul = df_ul[(df_ul["LogGFP"] > 1) & (df_ul["LogGFP"] < 4)]
    df_ul = stratified_sample(df_ul, "LogGFP", [500, 1000, 1500, 1500, 1000, 500])[["rep1", "rep2", "LogGFP"]]
    df_ul["Sequence"] = df_ul.index.map(UL)
    df_ul["Library"] = "UL"

    df_its = stratified_sample(pd.read_pickle(TABLE_DIR / "ITS.pkl"), "LogGFP", [500, 1000, 1500, 1500, 1000, 500])[["rep1", "rep2", "LogGFP"]]
    df_its["Sequence"] = df_its.index.map(ITS_context)
    df_its["Library"] = "ITS"

    return pd.concat([df_pl, df_sl16, df_sl17, df_sl18, df_dl, df_ul, df_its]).reset_index(drop=True)


class CorePromoterModel(nn.Module):
    def __init__(self, seq_length: int, num_conds: int):
        super().__init__()
        self.seq_length = seq_length
        self.conv1 = nn.Conv1d(4, 8, kernel_size=8, stride=1)
        self._initialize_conv1()
        self.conv2 = nn.Conv1d(8, 3, kernel_size=65, stride=1)
        self._initialize_conv2()
        self.conds_bias = nn.Parameter(torch.zeros(num_conds))
        self.param_max = nn.Parameter(torch.tensor(np.log(1000.0)))
        self.param_min = nn.Parameter(torch.tensor(np.log(0.5)))
        self.param_e0 = nn.Parameter(torch.tensor(0.0))

    def _initialize_conv1(self) -> None:
        motifs = ["AAAATTTG", "GAAAATAG", "TTGACATT", "NGGCCTAA", "ATGGGGTA", "TAATTTTT", "AAAGCAAA", "AAAAAGNN"]
        custom_weights = np.stack([getPFM(Motif2Seqs(motif), ratios=True).values.T for motif in motifs], axis=0)
        self.conv1.weight = nn.Parameter(torch.tensor(custom_weights, dtype=torch.float32), requires_grad=True)
        self.conv1.bias.data.zero_()
        self.conv1.bias.requires_grad = False

    def _initialize_conv2(self) -> None:
        f1 = np.zeros((8, 65))
        f2 = np.zeros((8, 65))
        f3 = np.zeros((8, 65))
        f1[0, 3], f1[1, 11], f1[2, 22], f1[3, 30], f1[4, 38], f1[5, 46], f1[6, 54], f1[7, 62] = 1, 1, 1, 1, 1, 1, 1, 1
        f2[0, 3], f2[1, 11], f2[2, 22], f2[3, 31], f2[4, 39], f2[5, 47], f2[6, 55], f2[7, 63] = 1, 1, 1, 1, 1, 1, 1, 1
        f3[0, 3], f3[1, 11], f3[2, 22], f3[3, 32], f3[4, 40], f3[5, 48], f3[6, 56], f3[7, 64] = 1, 1, 1, 1, 1, 1, 1, 1
        self.conv2.weight.data = torch.tensor(np.stack([f1, f2, f3], axis=0), dtype=torch.float32)
        self.conv2.weight.requires_grad = False
        self.conv2.bias.data.zero_()
        self.conv2.bias.requires_grad = True

    def architecture_logits(self, dna_seq: torch.Tensor) -> torch.Tensor:
        """Return every spacer-channel/position score before hard max pooling."""
        return self.conv2(self.conv1(dna_seq))

    def energy(self, dna_seq: torch.Tensor) -> torch.Tensor:
        x = self.architecture_logits(dna_seq)
        return torch.max(x.reshape(x.size(0), -1), dim=1, keepdim=True).values

    def energy2expression(self, x: torch.Tensor) -> torch.Tensor:
        beta = 1 / (0.001987 * 310)
        bw = torch.exp(beta * (x + self.param_e0))
        return torch.log(torch.exp(self.param_min) + torch.exp(self.param_max) * bw / (1 + bw))

    def forward(self, dna_seq: torch.Tensor, conds: torch.Tensor) -> torch.Tensor:
        x = self.energy(dna_seq)
        x = x + torch.matmul(conds, self.conds_bias.unsqueeze(1))
        return self.energy2expression(x)


def train_core_model(epochs: int = 50, batch_size: int = 512, device: torch.device | None = None) -> tuple[CorePromoterModel, pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    set_seed(SEED)
    df_train = build_core_training_dataframe()
    cat = pd.Categorical(df_train["Library"], categories=LIBRARY_ORDER)
    conds = pd.get_dummies(cat).to_numpy(dtype=np.float32)
    dna_seq = np.array(list(df_train["Sequence"].apply(dna_one_hot)), dtype=np.float32)
    target = (df_train["LogGFP"].values * np.log(10)).astype(np.float32)
    split = train_test_split(dna_seq, conds, target, test_size=0.2, random_state=42, stratify=None)
    xtr_np, xte_np, ctr_np, cte_np, ytr_np, yte_np = split
    xtr = torch.tensor(xtr_np, dtype=torch.float32, device=device)
    xte = torch.tensor(xte_np, dtype=torch.float32, device=device)
    ctr = torch.tensor(ctr_np, dtype=torch.float32, device=device)
    cte = torch.tensor(cte_np, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr_np, dtype=torch.float32, device=device).reshape(-1, 1)
    yte = torch.tensor(yte_np, dtype=torch.float32, device=device).reshape(-1, 1)
    loader = DataLoader(TensorDataset(xtr, ctr, ytr), batch_size=batch_size, shuffle=True)
    model = CorePromoterModel(seq_length=dna_seq.shape[2], num_conds=conds.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    crit = nn.MSELoss()
    hist = []
    for ep in range(epochs):
        model.train()
        for xb, cb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb, cb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            train_loss = crit(model(xtr, ctr) / np.log(10), ytr / np.log(10)).item()
            test_loss = crit(model(xte, cte) / np.log(10), yte / np.log(10)).item()
        hist.append({"epoch": ep + 1, "train_mse_log10": train_loss, "test_mse_log10": test_loss})
        print(f"Core epoch [{ep + 1:02d}/{epochs}] - Train {train_loss:.4f} Test {test_loss:.4f}")
    return model.eval(), pd.DataFrame(hist), df_train


class ElementEnergyModel(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 1, kernel_size=kernel_size, bias=False)
        self.param_max = nn.Parameter(torch.tensor(float(np.log(1000.0))))
        self.param_min = nn.Parameter(torch.tensor(float(np.log(0.5))))
        self.param_e0 = nn.Parameter(torch.tensor(0.0))

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h.view(h.size(0), -1)
        h = F.max_pool1d(h.unsqueeze(1), kernel_size=h.size(-1))
        return h.view(h.size(0))

    def energy2expression(self, e: torch.Tensor) -> torch.Tensor:
        beta = 1 / (0.001987 * 310)
        bw = torch.exp(beta * (e + self.param_e0))
        return torch.log(torch.exp(self.param_min) + torch.exp(self.param_max) * bw / (1 + bw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy2expression(self.energy(x)).reshape(-1, 1)


def correction_term(model: ElementEnergyModel) -> float:
    w = model.conv1.weight.detach().cpu()[0].numpy().T
    return float(w.mean(axis=1).sum())


def load_element_weight_model(weight_name: str, kernel_size: int, device: torch.device) -> ElementEnergyModel | None:
    path = WEIGHTS_DIR / weight_name
    if not path.exists():
        return None
    model = ElementEnergyModel(kernel_size).to(device)
    model.load_state_dict(torch_load_weights(path, device), strict=True)
    return model.eval()


def train_element_model(table_name: str, kernel_size: int, device: torch.device, epochs: int = 30, batch_size: int = 512, max_train_n: int = 20000) -> ElementEnergyModel:
    df = pd.read_pickle(TABLE_DIR / table_name).dropna(subset=["LogGFP"]).copy()
    df = df[df.index.to_series().astype(str).str.len() == kernel_size]
    df = df.sample(min(max_train_n, len(df)), random_state=SEED)
    x = np.array([dna_one_hot(s) for s in df.index], dtype=np.float32)
    y = (df["LogGFP"].values * np.log(10)).astype(np.float32)
    xtr_np, xte_np, ytr_np, yte_np = train_test_split(x, y, test_size=0.2, random_state=42)
    xtr = torch.tensor(xtr_np, dtype=torch.float32, device=device)
    xte = torch.tensor(xte_np, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr_np, dtype=torch.float32, device=device).reshape(-1, 1)
    yte = torch.tensor(yte_np, dtype=torch.float32, device=device).reshape(-1, 1)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True)
    model = ElementEnergyModel(kernel_size).to(device)
    opt = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    crit = nn.MSELoss()
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr = crit(model(xtr) / np.log(10), ytr / np.log(10)).item()
            te = crit(model(xte) / np.log(10), yte / np.log(10)).item()
        print(f"{table_name} epoch [{ep + 1:02d}/{epochs}] - Train {tr:.4f} Test {te:.4f}")
    return model.eval()


def score_energy_batched(model: ElementEnergyModel, seqs: list[str], device: torch.device, batch_size: int = 8192) -> np.ndarray:
    vals = []
    corr = correction_term(model)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start:start + batch_size]
            x = torch.tensor(np.array([dna_one_hot(s) for s in batch], dtype=np.float32), device=device)
            vals.append(model.energy(x).detach().cpu().numpy() - corr)
    return np.concatenate(vals)


def rank_strength_percentile(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
    n = len(out)
    out["rank_index"] = np.arange(n)
    out["strength_pct"] = 100.0 * (1.0 - out["rank_index"] / max(n - 1, 1))
    return out


def build_nn_pool(element: str, table_name: str, kernel_size: int, device: torch.device, weight_name: str | None = None, train_if_missing: bool = True, element_epochs: int = 30) -> pd.DataFrame:
    model = load_element_weight_model(weight_name, kernel_size, device) if weight_name else None
    if model is None:
        if not train_if_missing:
            raise FileNotFoundError(f"Missing element model weight: {weight_name}")
        model = train_element_model(table_name, kernel_size, device=device, epochs=element_epochs)
    raw = pd.read_pickle(TABLE_DIR / table_name).copy()
    raw = raw[raw.index.to_series().astype(str).str.len() == kernel_size]
    seqs = raw.index.astype(str).tolist()
    pool = pd.DataFrame({"element": element, "sequence": seqs})
    pool["score"] = score_energy_batched(model, seqs, device=device)
    if "LogGFP" in raw.columns:
        pool["LogGFP"] = raw.loc[seqs, "LogGFP"].values
    return rank_strength_percentile(pool, "score")


def build_bpm_pool(element: str) -> pd.DataFrame:
    all_hex = list(Motif2Seqs("N" * 6))
    spacer = "A" * 17
    ref_m35 = "TTGACA"
    ref_m10 = "TATAAT"
    if element == "m10":
        seqs = all_hex
        scores = [bpm_predict(ref_m35 + spacer + s) for s in seqs]
    elif element == "m35":
        seqs = all_hex
        scores = [bpm_predict(s + spacer + ref_m10) for s in seqs]
    else:
        raise ValueError(element)
    return rank_strength_percentile(pd.DataFrame({"element": element, "sequence": seqs, "score": scores}), "score")


def build_all_scored_pools(device: torch.device, element_epochs: int = 30) -> dict[str, pd.DataFrame]:
    return {
        "UP": build_nn_pool("UP", "UL.pkl", 19, device, "weights_UP.pt", element_epochs=element_epochs),
        "DIS": build_nn_pool("DIS", "DL.pkl", 8, device, "weights_Dis.pt", element_epochs=element_epochs),
        "ITS": build_nn_pool("ITS", "ITS.pkl", 10, device, None, element_epochs=element_epochs),
        "spacer": build_nn_pool("spacer", "SL17.pkl", 17, device, "weights_Sp17.pt", element_epochs=element_epochs),
        "m35": build_bpm_pool("m35"),
        "m10": build_bpm_pool("m10"),
    }


def nearest_band(pool: pd.DataFrame, target_pct: float, tolerance_pct: float) -> pd.DataFrame:
    sub = pool[(pool["strength_pct"] >= target_pct - tolerance_pct) & (pool["strength_pct"] <= target_pct + tolerance_pct)].copy()
    if sub.empty:
        sub = pool.copy()
    sub["target_pct"] = target_pct
    sub["pct_error"] = (sub["strength_pct"] - target_pct).abs()
    return sub.sort_values(["pct_error", "rank_index"]).reset_index(drop=True)


def build_slot_candidates_for_target(pool: pd.DataFrame, element: str, target_pct: float, slot_id: str, tolerance_pct: float, max_candidates: int, target_offsets: dict) -> pd.DataFrame:
    center = float(target_pct) + float(target_offsets.get((element, slot_id), target_offsets.get(element, 0.0)))
    sub = nearest_band(pool, center, tolerance_pct).copy()
    sub["element"] = element
    sub["slot_id"] = slot_id
    sub["target_pct"] = target_pct
    sub["center_pct"] = center
    sub["candidate_rank"] = np.arange(1, len(sub) + 1)
    return sub[["element", "slot_id", "target_pct", "center_pct", "candidate_rank", "sequence", "strength_pct", "pct_error", "score"]].head(max_candidates).reset_index(drop=True)


def build_spacer_tail_tg_candidates(spacer_pool: pd.DataFrame, spacer_40_candidates: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    base = spacer_40_candidates.copy().reset_index(drop=True)
    base["base_sequence"] = base["sequence"].astype(str)
    base["sequence"] = base["base_sequence"].str[:-2] + "TG"
    spacer_model = load_element_weight_model("weights_Sp17.pt", 17, device)
    if spacer_model is None:
        spacer_model = train_element_model("SL17.pkl", 17, device=device)
    base["score"] = score_energy_batched(spacer_model, base["sequence"].tolist(), device=device)
    ref = spacer_pool[["score"]].sort_values("score").reset_index(drop=True)
    base["strength_pct"] = [float((ref["score"] < val).mean() * 100.0) for val in base["score"]]
    base["element"] = "spacer"
    base["slot_id"] = "40_tailTG"
    base["target_pct"] = 40
    base["pct_error"] = (base["strength_pct"] - base["center_pct"]).abs()
    return base[["element", "slot_id", "target_pct", "center_pct", "candidate_rank", "sequence", "base_sequence", "strength_pct", "pct_error", "score"]]


def build_slot_candidates(scored_pools: dict[str, pd.DataFrame], device: torch.device, tolerance_pct: float, max_candidates: int, target_offsets: dict) -> dict[SlotKey, pd.DataFrame]:
    standard_targets = [10, 30, 50, 70, 90]
    spacer_targets = [20, 40, 60, 80]
    slots = {}
    for element in ["UP", "m35", "m10", "DIS", "ITS"]:
        for target in standard_targets:
            key = SlotKey(element, str(target))
            slots[key] = build_slot_candidates_for_target(scored_pools[element], element, target, str(target), tolerance_pct, max_candidates, target_offsets)
    for target in spacer_targets:
        key = SlotKey("spacer", str(target))
        slots[key] = build_slot_candidates_for_target(scored_pools["spacer"], "spacer", target, str(target), tolerance_pct, max_candidates, target_offsets)
    slots[SlotKey("spacer", "40_tailTG")] = build_spacer_tail_tg_candidates(
        scored_pools["spacer"],
        slots[SlotKey("spacer", "40")],
        device,
    )
    return slots


class RecursiveDesigner:
    def __init__(self, core_model: CorePromoterModel, slot_candidates: dict[SlotKey, pd.DataFrame], device: torch.device, scan_batch_size: int = 1024, success_mode: str = "anchor"):
        self.core_model = core_model
        self.slot_candidates = slot_candidates
        self.device = device
        self.scan_batch_size = scan_batch_size
        self.success_mode = success_mode

    def infer_model_m35_offset(self) -> int:
        conv2_w = self.core_model.conv2.weight.detach().cpu().numpy()
        offsets = []
        for ch in range(conv2_w.shape[0]):
            nz = np.where(np.abs(conv2_w[ch, 2]) > 1e-6)[0]
            if len(nz) != 1:
                raise ValueError(f"Cannot infer unique m35 offset for conv2 channel {ch}: {nz}")
            offsets.append(int(nz[0]))
        return int(np.median(offsets))

    def energy_to_log10(self, energy_tensor: torch.Tensor) -> np.ndarray:
        return (self.core_model.energy2expression(energy_tensor.reshape(-1, 1)) / np.log(10)).detach().cpu().numpy().ravel()

    def initial_state(self) -> dict[SlotKey, int]:
        return {key: 0 for key in self.slot_candidates}

    @staticmethod
    def replacement_key(key: SlotKey) -> SlotKey:
        if key.element == "spacer" and key.slot_id == "40_tailTG":
            return SlotKey("spacer", "40")
        return key

    def selected_elements_dataframe(self, state: dict[SlotKey, int]) -> pd.DataFrame:
        rows = []
        for key, idx in state.items():
            source_idx = state.get(SlotKey("spacer", "40"), idx) if key.element == "spacer" and key.slot_id == "40_tailTG" else idx
            cand = self.slot_candidates[key].iloc[int(source_idx)]
            rows.append({
                "element": key.element,
                "slot_id": key.slot_id,
                "sequence": cand["sequence"],
                "base_sequence": cand.get("base_sequence", ""),
                "target_pct": cand["target_pct"],
                "actual_pct": cand["strength_pct"],
                "candidate_rank": cand["candidate_rank"],
                "score": cand["score"],
            })
        return pd.DataFrame(rows).sort_values(["element", "target_pct", "slot_id"])

    def assemble_from_state(self, state: dict[SlotKey, int]) -> pd.DataFrame:
        selected = self.selected_elements_dataframe(state)
        elems = {element: selected.loc[selected["element"] == element, "sequence"].tolist() for element in ORDER}
        combos = list(itertools.product(*[elems[element] for element in ORDER]))
        df = pd.DataFrame(combos, columns=ORDER)
        df["Sequence"] = [BG5 + up + LINKER + m35 + sp + m10 + dis + its + BG3 for up, m35, sp, m10, dis, its in combos]
        df["design_m35_start"] = [len(BG5) + len(up) + len(LINKER) for up in df["UP"]]
        df["design_spacer_len"] = df["spacer"].str.len()
        df["design_m10_start"] = df["design_m35_start"] + M35_LEN + df["design_spacer_len"]
        df["design_channel"] = df["design_spacer_len"].map(CHANNEL_BY_SPACER)
        if df["design_channel"].isna().any():
            bad = sorted(df.loc[df["design_channel"].isna(), "design_spacer_len"].unique())
            raise ValueError(f"Unsupported spacer lengths: {bad}")
        df["design_channel"] = df["design_channel"].astype(int)
        df["model_m35_offset"] = self.infer_model_m35_offset()
        df["design_arch_start"] = df["design_m35_start"] - df["model_m35_offset"]
        return df

    def scan(self, df: pd.DataFrame) -> pd.DataFrame:
        self.core_model.eval()
        obs_arch, obs_ch, best_e, best_log, design_e, design_log = [], [], [], [], [], []
        with torch.no_grad():
            for start in range(0, len(df), self.scan_batch_size):
                sub = df.iloc[start:start + self.scan_batch_size]
                x = torch.tensor(np.array([dna_one_hot(seq) for seq in sub["Sequence"]], dtype=np.float32), dtype=torch.float32, device=self.device)
                conv2_out = self.core_model.conv2(self.core_model.conv1(x))
                n_channels = conv2_out.shape[1]
                n_pos = conv2_out.shape[2]
                flat = conv2_out.reshape(conv2_out.shape[0], -1)
                best_val, best_flat = flat.max(dim=1)
                best_ch = (best_flat // n_pos).detach().cpu().numpy().astype(int)
                best_pos = (best_flat % n_pos).detach().cpu().numpy().astype(int)
                obs_arch.extend(best_pos.tolist())
                obs_ch.extend(best_ch.tolist())
                best_e.extend(best_val.detach().cpu().numpy().tolist())
                best_log.extend(self.energy_to_log10(best_val).tolist())
                d_e = np.full(len(sub), np.nan, dtype=float)
                d_log = np.full(len(sub), np.nan, dtype=float)
                valid_idx, valid_vals = [], []
                for j, (_, row) in enumerate(sub.iterrows()):
                    ch = int(row["design_channel"])
                    pos = int(row["design_arch_start"])
                    if 0 <= ch < n_channels and 0 <= pos < n_pos:
                        valid_idx.append(j)
                        valid_vals.append(conv2_out[j, ch, pos])
                if valid_vals:
                    valid_tensor = torch.stack(valid_vals)
                    valid_energy = valid_tensor.detach().cpu().numpy()
                    valid_log = self.energy_to_log10(valid_tensor)
                    for j, e, logv in zip(valid_idx, valid_energy, valid_log):
                        d_e[j] = float(e)
                        d_log[j] = float(logv)
                design_e.extend(d_e.tolist())
                design_log.extend(d_log.tolist())
        out = df.copy()
        out["observed_arch_start"] = obs_arch
        out["observed_channel"] = obs_ch
        out["observed_spacer_len"] = out["observed_channel"].map(SPACER_BY_CHANNEL)
        out["arch_shift"] = out["observed_arch_start"] - out["design_arch_start"]
        out["channel_shift"] = out["observed_channel"] - out["design_channel"]
        out["observed_m35_start"] = out["observed_arch_start"] + out["model_m35_offset"]
        out["observed_m10_start"] = out["observed_m35_start"] + M35_LEN + out["observed_spacer_len"]
        out["m35_shift"] = out["observed_m35_start"] - out["design_m35_start"]
        out["m10_shift"] = out["observed_m10_start"] - out["design_m10_start"]
        out["best_model_energy"] = best_e
        out["best_model_log10"] = best_log
        out["design_model_energy"] = design_e
        out["design_model_log10"] = design_log
        out["delta_model_log10"] = out["best_model_log10"] - out["design_model_log10"]
        out["off_design"] = (out["arch_shift"].abs() > 2) | (out["channel_shift"] != 0)
        out["strong_off_design"] = out["off_design"] & (out["delta_model_log10"] > 0.3)
        out["clean_anchor"] = (out["m35_shift"] == 0) & (out["m10_shift"] == 0)
        out["clean_strict"] = out["clean_anchor"] & ~out["off_design"]
        return out

    def evaluate_state(self, state: dict[SlotKey, int]) -> pd.DataFrame:
        return self.scan(self.assemble_from_state(state))

    def success_mask(self, scan_df: pd.DataFrame) -> pd.Series:
        return scan_df["clean_strict"] if self.success_mode == "off_design" else scan_df["clean_anchor"]

    def summarize_scan(self, scan_df: pd.DataFrame, iteration: int, note: str = "") -> dict:
        clean = self.success_mask(scan_df)
        return {
            "iteration": iteration,
            "note": note,
            "n_designs": int(len(scan_df)),
            "clean_n": int(clean.sum()),
            "clean_rate": float(clean.mean()),
            "shifted_n": int((~clean).sum()),
            "m35_shift_rate": float((scan_df["m35_shift"] != 0).mean()),
            "m10_shift_rate": float((scan_df["m10_shift"] != 0).mean()),
            "both_shift_rate": float(((scan_df["m35_shift"] != 0) & (scan_df["m10_shift"] != 0)).mean()),
            "median_delta_model_log10": float(scan_df["delta_model_log10"].median()),
            "max_delta_model_log10": float(scan_df["delta_model_log10"].max()),
        }

    def slot_badness(self, scan_df: pd.DataFrame, state: dict[SlotKey, int]) -> pd.DataFrame:
        clean = self.success_mask(scan_df)
        selected = self.selected_elements_dataframe(state)
        rows = []
        for _, row in selected.iterrows():
            element = row["element"]
            seq = row["sequence"]
            mask = scan_df[element].astype(str) == str(seq)
            n = int(mask.sum())
            shifted = int((~clean[mask]).sum()) if n else 0
            rows.append({
                "element": element,
                "slot_id": str(row["slot_id"]),
                "sequence": seq,
                "target_pct": row["target_pct"],
                "actual_pct": row["actual_pct"],
                "candidate_rank": row["candidate_rank"],
                "n_contexts": n,
                "shifted_contexts": shifted,
                "bad_rate": shifted / n if n else np.nan,
            })
        return pd.DataFrame(rows).sort_values(["bad_rate", "shifted_contexts"], ascending=[False, False]).reset_index(drop=True)

    def available_indices(self, key: SlotKey, current_idx: int, tried: set[int], limit: int) -> list[int]:
        out = []
        for idx in range(current_idx + 1, len(self.slot_candidates[key])):
            if idx not in tried:
                out.append(idx)
            if len(out) >= limit:
                break
        return out

    def probe_best_replacement(self, state: dict[SlotKey, int], key: SlotKey, tried: dict[SlotKey, set[int]], limit: int):
        current_idx = state[key]
        candidate_indices = self.available_indices(key, current_idx, tried.setdefault(key, {current_idx}), limit)
        best_idx, best_scan, best_summary = None, None, None
        for idx in candidate_indices:
            temp = dict(state)
            temp[key] = idx
            scanned = self.evaluate_state(temp)
            summary = self.summarize_scan(scanned, -1, f"probe {key.element}:{key.slot_id} idx={idx}")
            summary["probe_element"] = key.element
            summary["probe_slot_id"] = key.slot_id
            summary["probe_candidate_idx"] = idx
            better = best_summary is None or summary["clean_rate"] > best_summary["clean_rate"]
            tied_better = best_summary is not None and summary["clean_rate"] == best_summary["clean_rate"] and summary["shifted_n"] < best_summary["shifted_n"]
            if better or tied_better:
                best_idx, best_scan, best_summary = idx, scanned, summary
        return best_idx, best_scan, best_summary


def export_outputs(out_dir: Path, designer: RecursiveDesigner, best_state, best_scan, history_df, final_badness, success: bool) -> None:
    final_elements = designer.selected_elements_dataframe(best_state)
    final_elements.to_csv(out_dir / "recursive_elements_shared.csv", index=False)
    best_scan.to_csv(out_dir / "recursive_corepromoter_scan.csv", index=False)
    history_df.to_csv(out_dir / "recursive_search_history.csv", index=False)
    final_badness.to_csv(out_dir / "recursive_final_slot_badness.csv", index=False)
    with pd.ExcelWriter(out_dir / "recursive_design_summary.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([{"success": success, "success_mode": designer.success_mode, "output_dir": str(out_dir)}]).to_excel(writer, "Run_Status", index=False)
        history_df.to_excel(writer, "Search_History", index=False)
        final_elements.to_excel(writer, "Final_Elements", index=False)
        final_badness.to_excel(writer, "Final_Slot_Badness", index=False)
        best_scan.sort_values(["clean_anchor", "delta_model_log10"], ascending=[True, False]).head(200).to_excel(writer, "Worst_Examples", index=False)


def run_search_only(
    core_model: CorePromoterModel,
    slot_candidates: dict[SlotKey, pd.DataFrame],
    out_dir: Path,
    device: torch.device | None = None,
    max_search_iters: int = 30,
    probe_candidates_per_slot: int = 6,
    n_worst_slots_per_iter: int = 1,
    scan_batch_size: int = 1024,
    success_mode: str = "anchor",
) -> RecursiveDesignResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    out_dir.mkdir(parents=True, exist_ok=True)

    designer = RecursiveDesigner(core_model, slot_candidates, device, scan_batch_size=scan_batch_size, success_mode=success_mode)
    state = designer.initial_state()
    tried = {key: {idx} for key, idx in state.items()}
    history, probe_history = [], []

    scan_df = designer.evaluate_state(state)
    summary = designer.summarize_scan(scan_df, 0, "initial")
    history.append(summary)
    best_state, best_scan, best_summary = dict(state), scan_df.copy(), summary
    success = summary["shifted_n"] == 0
    print(summary)

    for iteration in range(1, max_search_iters + 1):
        if success:
            break
        badness = designer.slot_badness(scan_df, state)
        badness.to_csv(out_dir / f"slot_badness_iter_{iteration:02d}.csv", index=False)
        candidate_slots = []
        for _, row in badness.head(n_worst_slots_per_iter).iterrows():
            key = designer.replacement_key(SlotKey(row["element"], str(row["slot_id"])))
            if key not in candidate_slots:
                candidate_slots.append(key)

        best_move, best_move_scan, best_move_summary = None, None, None
        for key in candidate_slots:
            idx, probed_scan, probed_summary = designer.probe_best_replacement(state, key, tried, probe_candidates_per_slot)
            if probed_summary is None:
                continue
            probe_history.append(probed_summary)
            better = best_move_summary is None or probed_summary["clean_rate"] > best_move_summary["clean_rate"]
            tied_better = best_move_summary is not None and probed_summary["clean_rate"] == best_move_summary["clean_rate"] and probed_summary["shifted_n"] < best_move_summary["shifted_n"]
            if better or tied_better:
                best_move, best_move_scan, best_move_summary = (key, idx), probed_scan, probed_summary
        if best_move is None:
            print("No available replacements left for the worst slots.")
            break

        move_key, move_idx = best_move
        state[move_key] = move_idx
        tried.setdefault(move_key, set()).add(move_idx)
        scan_df = best_move_scan
        summary = designer.summarize_scan(scan_df, iteration, f"replace {move_key.element}:{move_key.slot_id} -> candidate_idx {move_idx}")
        history.append(summary)
        print(summary)

        better_best = summary["clean_rate"] > best_summary["clean_rate"]
        tied_best = summary["clean_rate"] == best_summary["clean_rate"] and summary["shifted_n"] < best_summary["shifted_n"]
        if better_best or tied_best:
            best_state, best_scan, best_summary = dict(state), scan_df.copy(), summary
        success = summary["shifted_n"] == 0

    history_df = pd.DataFrame(history)
    probe_history_df = pd.DataFrame(probe_history)
    final_badness = designer.slot_badness(best_scan, best_state)
    probe_history_df.to_csv(out_dir / "recursive_probe_history.csv", index=False)
    export_outputs(out_dir, designer, best_state, best_scan, history_df, final_badness, success)
    return RecursiveDesignResult(
        success=success,
        best_summary=best_summary,
        out_dir=out_dir,
        history=history_df,
        final_elements=designer.selected_elements_dataframe(best_state),
        final_badness=final_badness,
        best_scan=best_scan,
    )


def run_recursive_design(
    core_epochs: int = 50,
    element_epochs: int = 30,
    target_tolerance_pct: float = 5.0,
    max_candidates_per_slot: int = 120,
    max_search_iters: int = 30,
    probe_candidates_per_slot: int = 6,
    n_worst_slots_per_iter: int = 1,
    scan_batch_size: int = 1024,
    success_mode: str = "anchor",
    target_offsets: dict | None = None,
    parent_out_dir: Path = DEFAULT_PARENT_OUT,
) -> RecursiveDesignResult:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = parent_out_dir / f"recursive_design_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", device)
    print("Output:", out_dir)

    core_model, core_history, core_training_df = train_core_model(core_epochs, device=device)
    core_history.to_csv(out_dir / "core_training_history.csv", index=False)
    core_training_df.groupby("Library").size().rename("n").to_csv(out_dir / "core_training_counts.csv")

    scored_pools = build_all_scored_pools(device=device, element_epochs=element_epochs)
    for element, pool in scored_pools.items():
        pool.to_csv(out_dir / f"pool_{element}.csv", index=False)

    slot_candidates = build_slot_candidates(
        scored_pools,
        device=device,
        tolerance_pct=target_tolerance_pct,
        max_candidates=max_candidates_per_slot,
        target_offsets=target_offsets or {},
    )
    slot_summary = pd.DataFrame([
        {
            "element": key.element,
            "slot_id": key.slot_id,
            "n_candidates": len(cands),
            "first_seq": cands.iloc[0]["sequence"] if len(cands) else "",
            "first_strength_pct": cands.iloc[0]["strength_pct"] if len(cands) else np.nan,
        }
        for key, cands in slot_candidates.items()
    ])
    slot_summary.to_csv(out_dir / "slot_candidate_summary.csv", index=False)

    return run_search_only(
        core_model=core_model,
        slot_candidates=slot_candidates,
        out_dir=out_dir,
        device=device,
        max_search_iters=max_search_iters,
        probe_candidates_per_slot=probe_candidates_per_slot,
        n_worst_slots_per_iter=n_worst_slots_per_iter,
        scan_batch_size=scan_batch_size,
        success_mode=success_mode,
    )
