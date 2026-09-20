# Promoter Library Design

以 PyTorch 建立 promoter expression scoring models，並利用 whole/CorePromoter model、個別 promoter element models 與 automated search 設計 promoter library。

本專案目前包含：

- baseline whole/CorePromoter model 的訓練流程與 frozen checkpoint
- UP、-35、spacer、-10、DIS、ITS 等 element scoring models
- 以 energy bin 從 high-throughput database 取樣的 automated promoter library redesign
- CorePromoter model energy 與天然 promoter 保守度的關聯分析
- 組裝後整條序列的 restriction-site scan

## Pipeline 概覽

```text
Expression libraries (PL / SL16 / SL17 / SL18 / DL / UL / ITS)
        │
        ├──► Model_CorePromoter_clean.ipynb ──► weights_CorePromoter_clean.pt
        │        (whole model 訓練與 filter logo)        [frozen baseline]
        │
        └──► Model_UP / Model_PL / Model_Sp16-18 /
             Model_Dis / Model_ITS.ipynb  ──────────────► weights_UP / Sp16-18 /
                                                          Dis / ITS.pt、BPM 參數
                                │
        Element models + energy bins
                                │
                                ▼
        library_release/01_recursive_design.ipynb
        (automated_promoter_library_design/)
                                │
                                ▼
                    promoter library outputs

weights_CorePromoter_clean.pt ──► Model_CorePromoter_energy_vs_conservation.ipynb
                                  (energy vs IC / KL / log2 enrichment)
```

訓練與 library design 是分開的步驟。checkpoint 儲存後，design 只會載入指定權重進行評分與搜尋，不會自動重新訓練模型。

## 主要目錄

```text
MS2_Data_PyTorch/
├── scripts/
│   ├── Model_CorePromoter_clean.ipynb              # baseline whole model 訓練 + filter logo
│   ├── Model_CorePromoter_v0.ipynb                 # clean 版之前的完整原始 notebook（保留參考）
│   ├── Model_CorePromoter_energy_vs_conservation.ipynb  # energy vs 天然保守度分析
│   ├── Model_UP.ipynb / Model_Sp16.ipynb / Model_Sp17.ipynb / Model_Sp18.ipynb
│   │   / Model_Dis.ipynb / Model_ITS.ipynb
│   │                                               # element scoring models
│   ├── Model_PL.ipynb                              # -35 / -10 的 BPM data flow
│   ├── automated_promoter_library_design/          # 設計函式庫，一個模組對應一個 Batch
│   │   ├── __init__.py                         # 門面與索引：整包的公開名稱都在這
│   │   ├── _common.py                          # 常數與小工具（各 Batch 都會用到）
│   │   ├── config.py                           # DesignConfig 與 run config      → Batch 0
│   │   ├── scoring.py                          # 元件打分、序列池、能量分箱       → Batch 0.5、1
│   │   ├── space.py                            # DesignSpace、gap 分配、組裝      → Batch 2、3
│   │   ├── scanning.py                         # register 掃描與偏移診斷          → Batch 4、5
│   │   └── search.py                           # AutomatedRedesigner              → Batch 6
│   ├── recursive_corepromoter_design.py            # 共用 model 定義與訓練資料組裝
│   ├── util.py                                     # PFM / logo / 繪圖小工具
│   ├── BPM/                                        # BPM.py、util.py、Params_Con17.pkl
│   └── library_release/                            # 文庫產出 pipeline 01-07，見該目錄 README
├── tables/      # 本機 experimental data，除少數小檔外不納入 Git
├── genomes/     # 參考基因體 GenBank 檔（*.gb，不納入 Git）
├── weights/     # 已訓練的 checkpoint 與 metadata（納入 Git）
├── figures/     # notebook 產生的圖（*.png / *.svg，不納入 Git）
└── outputs/     # energy_vs_conservation 的輸出（不納入 Git）

outputs/design_runs/        # library design 每次搜尋的輸出（不納入 Git）
outputs/energy_bin_cache/   # scored pool 快取（不納入 Git）
```

## Environment

`requirements.txt` 列出主要 dependencies，但沒有鎖定版本：

```text
torch, numpy, pandas, scipy, scikit-learn, openpyxl, jupyterlab, matplotlib, seaborn
```

下列套件目前**沒有**寫進 `requirements.txt`，但實際會用到，請自行補裝：

| 套件 | 用在哪裡 |
| --- | --- |
| `biopython` | `Model_CorePromoter_energy_vs_conservation.ipynb` 讀 GenBank |
| `logomaker` | `Model_CorePromoter_clean.ipynb` 與各 element model 的 sequence logo |
| `tensorboard` | `Model_CorePromoter_clean.ipynb` / `Model_CorePromoter_v0.ipynb` 的 `SummaryWriter` |
| `tqdm` | `automated_promoter_library_design/search.py` 的進度顯示（缺少時自動退回無進度條） |

建議建立獨立 virtual environment 後再安裝。GPU 並非必要；CLI scripts 的 `--device auto` 會在 CUDA 可用時使用 GPU，notebook 則以 `torch.cuda.is_available()` 自動選擇。`Model_CorePromoter_energy_vs_conservation.ipynb` 固定使用 CPU。

## Local data requirements

大型 experimental tables 不納入 Git，必須自行放在 `MS2_Data_PyTorch/tables/`：

```text
PL.pkl      # -35 / -10（Model_PL、BPM）
SL16.pkl    # spacer 16
SL17.pkl    # spacer 17
SL18.pkl    # spacer 18
DL.pkl      # DIS
UL.pkl      # UP
ITS.pkl     # ITS
```

其他資料需求：

- 保守度分析與 `library_release/02`：`MS2_Data_PyTorch/tables/Data_S1_20250826.xlsx`
- 保守度分析的背景組成：`MS2_Data_PyTorch/genomes/NC_000913.2.gb`（E. coli K-12 MG1655）；`*.gb` 被 gitignore，需自行下載
- `library_release/07_re_scan.ipynb`：讀 `library_release/outputs/06_whole_sequence.csv`
  （舊的 `tables/assembled_scan.csv` 是 93 nt 的過期 BG5/BG3 版本，已移到 `_archive_20260918/`）
- repository 內只保留兩份小型 shared tables：`elements_shared.csv`、`phage_promoters.csv`

> Experimental tables 含本機資料，Git clone 後不會自動取得。執行前請先確認檔名、欄位與路徑符合 scripts 的預期。

## 1. Baseline whole model

開啟並依序執行：

```text
MS2_Data_PyTorch/scripts/Model_CorePromoter_clean.ipynb
```

這本 notebook 做的是：組裝 7 個 library 的訓練資料 → 定義 8 filters × 3 spacer channels 的 `DNAFunctionPredictor` → train/test 評估 → 畫 conv1 filter logo 與 spacer 對齊後的 spatial logo。

**它不會寫出 checkpoint。** `weights/weights_CorePromoter_clean.pt` 是已納入 Git 的 frozen baseline，下游（library design、保守度分析）一律載入這個檔案。要更新 baseline 必須手動存檔並確認下游是否需要重跑。相關的 `_history.csv`、`_metadata.json`、`_training_counts.csv` 同樣保留在 `weights/`。

`Model_CorePromoter_v0.ipynb` 是 clean 版之前的完整 notebook，含 BPM scan 與 debug cells，僅作參考。

## 2. Element scoring models

| Notebook | 來源 table | 產出 |
| --- | --- | --- |
| `Model_UP.ipynb` | `UL.pkl` | `weights_UP.pt` |
| `Model_Sp16.ipynb` / `Model_Sp17.ipynb` / `Model_Sp18.ipynb` | `SL16.pkl` / `SL17.pkl` / `SL18.pkl` | `weights_Sp16.pt` / `Sp17` / `Sp18`。只有 Sp17 參與打分，但三個檔缺一不可，見 `weights/README.md` |
| `Model_Dis.ipynb` | `DL.pkl` | `weights_Dis.pt` |
| `Model_ITS.ipynb` | `ITS.pkl` | `weights_ITS.pt` |
| `Model_PL.ipynb` | `PL.pkl` | BPM 參數（`BPM/Params_Con17.pkl`） |

**-35 / -10 使用 BPM，不使用 NN weights。** `weights_minus35_unused.pt` 與 `weights_minus10_unused.pt` 雖然存在 `weights/`，但 pipeline 沒有任何 code 載入它們（`ElementModelBundle._load_all` 明確跳過），而且與 BPM 排序嚴重不一致（Spearman ρ = 0.37 / −0.22）。design 流程一律以 `-BPM dG` 作為 -35/-10 的 higher-is-stronger score。

所有 element score 都是「higher = stronger」，但**不同 element 的分數不可互相比較**。

## 3. Automated promoter library redesign

主要入口：

```text
MS2_Data_PyTorch/scripts/library_release/01_recursive_design.ipynb
   └── 實作在 automated_promoter_library_design/（一個模組對應一個 Batch，見該套件的 __init__.py）
```

這個流程**從 high-throughput PKL database 挑選既有序列，不做 de novo generation**。每個 element 依自己的 energy 分布切成等寬 bin，每個 bin 出一個 mutable 版本，再加一個 locked consensus；所有 mutable 版本必須弱於 locked consensus。

### Notebook batch 結構

| Batch | 內容 |
| --- | --- |
| 0 | 設定：checkpoint、RUN_MODE、consensus、energy bin 邊界、search settings |
| 0.5 | 六個 library 的 energy 分布圖與現行 pooling 邊界（用來決定 Batch 0 的數字） |
| 1 | 載入 element models 與 core model，建立 scored pools 與 bin summary |
| 2 | 建立 design units 並執行 hard constraints |
| 3 | 固定且平均分配的 3-bp gap assignment 與 library 組裝 |
| 4 | 初始 CorePromoter register scan 與 validation |
| 5 | dominant shift 與 conditional-risk 診斷 |
| 6 | 單調式自動 redesign 搜尋（可中斷、可 resume） |
| 7 | 最終設計與 audit tables |

### 主要設定（Batch 0）

```python
CORE_MODEL_CHECKPOINT = r.WEIGHTS_DIR / "weights_CorePromoter_clean.pt"
RUN_MODE = "new"                 # "new" or "resume"
RESUME_DIR = r.DEFAULT_PARENT_OUT / "automated_redesign_<timestamp>"
```

energy bin 邊界逐 element 指定為 `(lower_fraction, upper_fraction, n_bins)`，數值是該 element 自己觀測 min–max energy 軸上的比例，可直接從 Batch 0.5 的上緣副軸讀出：

```python
ENERGY_BIN_RANGES = {
    # UP 只切 3 個 bin：它的 locked consensus 有兩條，佔掉 v4 與 v5，
    # 所以它仍然是 5 個版本，整個文庫維持 5^6 = 15,625。
    "UP":     (0.0, 0.7, 3),
    "m35":    (0.2, 1.0, 4),
    "spacer": (0.0, 0.9, 4),
    "m10":    (0.0, 1.0, 4),
    "DIS":    (0.0, 1.0, 4),
    "ITS":    (0.0, 1.0, 4),
}
```

- N 個 bin → mutable `v1..vN` + locked consensus 填滿其上的版本，一個 design state 組成 `∏(版本數)` 個 variants。目前六個元素都是 5 個版本，所以是 `5^6 = 15,625`：m35 / spacer / m10 / DIS / ITS 各 4 個 bin 加 1 個 locked，UP 則是 3 個 bin 加 2 條 locked consensus。
- spacer 至少需要 3 個 bin：`Spacer_v3 = Spacer_v2[:-2] + "TG"` 是衍生的，不取自 bin 3，v2/v3 是同一個 coupled design unit。
- 某個 bin 的上界超過 locked consensus 時該 bin 沒有 eligible sequence，`DesignSpace` 會直接報錯而不是跨 bin 補候選 —— 這時回 Batch 0 調低 `upper_fraction`。bin summary 的 `n_eligible_below_v5` 就是扣掉這個限制後真正可用的候選數。

search 參數：

```python
SEARCH_SETTINGS = {
    "max_iterations": 50,         # 每次 new/resume 執行的「額外」輪數，不是累積上限
    "n_driver_units": 3,
    "probe_candidates_per_unit": 2,
    "pair_beam_width": 6,
    "max_pair_evaluations": 8,
    "max_stalled_iterations": 60,
}
```

`redesigner.run(..., verbose=False)` 可設為 `True` 取得每個 proposal 的逐項輸出；`progress_df` 會在記憶體中每輪原地更新，手動中斷後仍可 `display(progress_df.tail())`。

### Validation 規則

```text
m10_shift = observed_m10_start - design_m10_start
m35_shift = observed_m35_start - design_m35_start
```

- `shift != 0` 就計入 shifted variant。
- m10 與 m35 的 shifted rate 都必須 `< 10%`（`max_shift_rate`）。
- 所有 shifted variants 必須落在 `-2..+2 bp`（`max_abs_shift`）。
- 先最佳化 m10；m10 通過後轉為 hard constraint，再最佳化 m35。
- 只有 global validation objective 嚴格改善才接受 replacement。

### 輸出

執行期間持續覆寫（可據此 resume）：

```text
search_progress.csv
proposal_history_checkpoint.csv
current_elements_checkpoint.csv
current_validation.json
search_checkpoint.json
```

結束後另外寫出：

```text
run_config.json / resume_config_<timestamp>.json
energy_bin_summary.csv
design_unit_candidate_counts.csv
gap_assignment.csv
initial_elements.csv / initial_assembled_<N>.csv / initial_scan_<N>.csv / initial_validation.json
final_elements.csv / final_scan_<N>.csv / final_validation.json
search_history.csv / proposal_history.csv
final_driver_risk.csv / final_overlap_evidence.csv
risk_iter_<NN>.csv / overlap_iter_<NN>.csv / accepted_elements_iter_<NN>.csv
```

預設輸出位置：

```text
outputs/design_runs/automated_redesign_<timestamp>/
outputs/energy_bin_cache/            # scored pool 快取，依 PKL 與 weight 的檔案 signature 失效
```

改動 bin 邊界不需要重算 energy，快取仍然有效；來源 PKL 或 element weights 更新後會自動重建。

## 4. Energy vs conservation 分析

```text
MS2_Data_PyTorch/scripts/Model_CorePromoter_energy_vs_conservation.ipynb
```

把 `weights_CorePromoter_clean.pt` 拆成以 -10 為基準的 4 × L energy matrix，與真實 E. coli promoter 的位置保守度做關聯。因為 `conv2` 只是把 conv1 的 8 個 8-bp filter 擺在固定位移上相加，整個模型的 energy 嚴格可加，可以無損拆解：

```text
E(sequence) = Σ_i e(i, base_i) + conv2.bias[channel]
```

主要設定：

```python
SPACER = 17                     # 只分析 spacer=17（conv2 channel 1）
SHEETS = ["Es.co", "Es.co$"]
BACKGROUND_SOURCE = "genome"    # "genome" 或 "flank"
CORE_RANGE = (-23, 9)
RETRAIN = False                 # True = 依 clean notebook 流程重訓一次（需要 tables 裡的 pkl）
```

統計方法：Pearson r（含 5000 次 bootstrap 百分位 95% CI）、Spearman ρ（p 值改用 permutation test，n! 夠小時為 exact test），另有 `r_between` / `r_within` 拆解，用來區分「純鹼基組成偏好」與「位置專一資訊」。

輸出：

```text
MS2_Data_PyTorch/outputs/energy_vs_conservation_Ecoli_sp17_genomebg_positions.csv
MS2_Data_PyTorch/outputs/energy_vs_conservation_Ecoli_sp17_genomebg_perbase.csv
MS2_Data_PyTorch/outputs/energy_vs_conservation_Ecoli_sp17_genomebg_correlations.csv
MS2_Data_PyTorch/figures/EnergyVsIC_position_byElement_Ecoli_sp17.{png,svg}
MS2_Data_PyTorch/figures/EnergyVsKL_position_byElement_Ecoli_sp17.{png,svg}
MS2_Data_PyTorch/figures/EnergyVsLog2Enrichment_perbase_byElement_Ecoli_sp17.{png,svg}
```

`plot_by_element(..., groups=[...])` 可把欄位換成任意位置子集（元件名、rel_pos 區間、`element_positions()` 取前/後 N 個、或自訂遮罩），版面與統計標註完全一樣；notebook 內附五個已註解的範例。

## 5. Restriction site scan

```text
MS2_Data_PyTorch/scripts/library_release/07_re_scan.ipynb
```

對整條 `full_sequence` 掃 19 種常用 restriction site（非回文的 Eco31I 連反股一起掃）。利用 06 記錄的 segment offset，把每個切位歸屬到 BG5 / promoter / RE1 / RE2 / barcode / BG3，只有落在固定側翼之外的才算 unexpected。輸出 `outputs/07_re_scan_report.csv`。

## Model checkpoint 說明

| Checkpoint | 用途 | 由誰產生 |
| --- | --- | --- |
| `weights_CorePromoter_clean.pt` | baseline whole model；design 與保守度分析的預設 | frozen，已納入 Git |
| `weights_UP / Sp16 / Sp17 / Sp18 / Dis / ITS.pt` | element energy models | 對應的 element notebook |
| `weights_minus35_unused.pt` / `weights_minus10_unused.pt` | **未使用**，-35/-10 走 BPM | 來源不明，產生它們的程式不在 repo；見 `weights/README.md` |

- `.pt` 檔是已訓練的 model parameters/checkpoint；checkpoint 存在時，推論與 library design 不需要重新訓練。
- 只有改變 model architecture、training data、training objective，或需要重新估計 weights 時才需重訓。
- design 結果依賴所選 checkpoint、element models、輸入 tables 與 search configuration，這些都會記在 `run_config.json`。

## Reproducibility notes

- 主要 scripts 使用固定 random seed `777`（gap assignment 另有 `gap_seed`），但不同 PyTorch、CUDA 或硬體環境仍可能造成小幅差異。
- 每次 run 應保留對應的 `run_config.json` / `training_config.json` 與 model metadata；`run_config.json` 內含 core checkpoint 的檔案 signature（路徑、大小、mtime）。
- 不要將大型 raw data、generated outputs、secrets 或暫存檔 commit 至 Git。
- Jupyter notebooks 在 commit 前建議清除不必要的 cell outputs，以減少 repository 大小並避免留下本機路徑或大量執行結果。

## Git workflow

修改完成後先檢查差異，再自行 Stage、Commit 與 Push：

```powershell
git status
git diff
git add README.md
git commit -m "Update project README"
git push
```

`git add README.md` 只會 Stage README，不會把其他 untracked files 一起加入 commit。
