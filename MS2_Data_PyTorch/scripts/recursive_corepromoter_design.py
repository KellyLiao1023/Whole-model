from __future__ import annotations

import random
import sys
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
SPACERS = [16, 17, 18]
SPACER_BY_CHANNEL = {ch: sp for ch, sp in enumerate(SPACERS)}
CHANNEL_BY_SPACER = {sp: ch for ch, sp in SPACER_BY_CHANNEL.items()}


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
        # Padding means "no sequence here", so it must contribute nothing. The
        # previous [-1,-1,-1,-1] made conv1's output at that position the NEGATED
        # column sum, and nothing constrained that sum to stay positive: training
        # could lower it and turn padding into a score bonus. Only UL() carries
        # dashes, and UL is the one library whose variable region sits under conv1
        # filter 1, so the model could cut UL's loss by reading padding instead of
        # learning UP -- then those distorted weights scored designed sequences that
        # contain no dashes at all. Measured before the fix: every one of the eight
        # filters gave a positive contribution for an all-dash window.
        # [0,0,0,0], not [0.25]*4: "absent", not "any base".
        "-": [0, 0, 0, 0],
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
