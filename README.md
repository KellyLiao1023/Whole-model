# Promoter Library Design

以 PyTorch 建立 promoter expression scoring models，並利用 whole/CorePromoter model、個別 promoter element models 與 recursive search 設計 promoter library。

本專案目前包含：

- baseline whole/CorePromoter model 訓練與權重
- 使用多物種 TSS/PAS 資料改良 CorePromoter architecture 辨識能力
- UP、-35、spacer、-10、DIS、ITS 等 element scoring models
- recursive promoter library design 與 restriction-site / sequence QC
- baseline 與 TSS/PAS CorePromoter model 的比較評估

## Pipeline 概覽

```text
Expression libraries
        │
        ▼
Model_CorePromoter_clean.ipynb
        │
        └── weights_CorePromoter_clean.pt  (baseline whole model)
                        │
TSS/PAS workbook ──────┤
                        ▼
Model_CorePromoter_TSS_pretrain.ipynb
或 train_corepromoter_tss_pas.py
                        │
                        ├── weights_CorePromoter_tss_arch.pt
                        └── weights_CorePromoter_tss_pas.pt
                                      │
Element models + sequence tables ─────┤
                                      ▼
Model_CorePromoter_recursive_design.ipynb
                                      │
                                      ▼
                         promoter library outputs
```

訓練與 recursive design 是分開的步驟。模型完成訓練並儲存 checkpoint 後，recursive design 只會載入指定權重進行評分與搜尋，不會自動重新訓練模型。

## 主要目錄

```text
MS2_Data_PyTorch/
├── scripts/
│   ├── Model_CorePromoter_clean.ipynb
│   ├── Model_CorePromoter_TSS_pretrain.ipynb
│   ├── Model_CorePromoter_recursive_design.ipynb
│   ├── Model_UP_v0.ipynb
│   ├── Model_PL.ipynb
│   ├── Model_Sp17.ipynb
│   ├── Model_Dis.ipynb
│   ├── Model_ITS.ipynb
│   ├── train_corepromoter_tss_pas.py
│   ├── evaluate_corepromoter_tss_pas.py
│   ├── tss_pas_dataset.py
│   ├── recursive_corepromoter_design.py
│   ├── automated_promoter_library_design.py
│   ├── pas_library_qc.py
│   └── BPM/
├── tables/                 # local experimental data; mostly excluded from Git
└── weights/                # trained model checkpoints and metadata

outputs/                    # generated run results; excluded from Git
Promoter_library_design/    # downstream selection and QC tools
```

## Environment

目前 repository 沒有鎖定版本的 `requirements.txt` 或 environment file。主要 Python dependencies 包含：

- Python 3
- PyTorch
- NumPy
- pandas
- SciPy
- scikit-learn
- openpyxl
- Jupyter

建議建立獨立 virtual environment 後再安裝 dependencies。GPU 並非必要；scripts 的 `--device auto` 會在 CUDA 可用時使用 GPU，否則使用 CPU。

## Local data requirements

大型 experimental tables 不納入 Git，必須自行放在：

```text
MS2_Data_PyTorch/tables/
```

whole model 與 element models 會使用下列 pickle tables：

```text
PL.pkl
SL16.pkl
SL17.pkl
SL18.pkl
DL.pkl
UL.pkl
ITS.pkl
```

TSS/PAS training 預設使用：

```text
MS2_Data_PyTorch/tables/Data_S1_20250826.xlsx
```

部分 design 流程也會使用 repository 內保留的小型 shared tables，例如 `elements_shared.csv` 與 `phage_promoters.csv`。

> Experimental tables 含本機資料，Git clone 後不會自動取得。執行前請先確認檔名、欄位與路徑符合 scripts 的預期。

## 1. Baseline whole model

開啟並依序執行：

```text
MS2_Data_PyTorch/scripts/Model_CorePromoter_clean.ipynb
```

完成後產生的主要 checkpoint 為：

```text
MS2_Data_PyTorch/weights/weights_CorePromoter_clean.pt
```

相關 training history、metadata 與 training counts 也儲存在 `MS2_Data_PyTorch/weights/`。

## 2. TSS/PAS CorePromoter training

### Notebook 方式

開啟並依序執行：

```text
MS2_Data_PyTorch/scripts/Model_CorePromoter_TSS_pretrain.ipynb
```

### Command line 方式

從 repository 根目錄執行：

```powershell
python MS2_Data_PyTorch/scripts/train_corepromoter_tss_pas.py --prepare-data --device auto
```

訓練分為三個階段：

1. TSS/PAS architecture pretraining
2. expression head recalibration
3. architecture 與 expression joint fine-tuning

主要輸出：

```text
MS2_Data_PyTorch/weights/weights_CorePromoter_tss_arch.pt
MS2_Data_PyTorch/weights/weights_CorePromoter_tss_pas.pt
outputs/corepromoter_tss_pas/<timestamp>/training_history.csv
outputs/corepromoter_tss_pas/<timestamp>/model_metrics.csv
outputs/corepromoter_tss_pas/<timestamp>/training_config.json
```

若 processed dataset 已存在，可以省略 `--prepare-data`。常用選項可用以下指令查看：

```powershell
python MS2_Data_PyTorch/scripts/train_corepromoter_tss_pas.py --help
```

## 3. Model evaluation

比較 baseline 與 TSS/PAS checkpoints：

```powershell
python MS2_Data_PyTorch/scripts/evaluate_corepromoter_tss_pas.py --device auto
```

若需要較耗時的 leave-one-library-out comparison：

```powershell
python MS2_Data_PyTorch/scripts/evaluate_corepromoter_tss_pas.py --device auto --run-lolo
```

評估結果預設輸出至：

```text
outputs/corepromoter_tss_pas_evaluation/<timestamp>/
```

## 4. Recursive promoter design

開啟：

```text
MS2_Data_PyTorch/scripts/Model_CorePromoter_recursive_design.ipynb
```

在設定 cell 選擇 CorePromoter scoring model：

```python
CORE_MODEL_VARIANT = "tss_pas"  # "baseline" or "tss_pas"
```

對應 checkpoint：

| Variant | Checkpoint |
| --- | --- |
| `baseline` | `weights_CorePromoter_clean.pt` |
| `tss_pas` | `weights_CorePromoter_tss_pas.pt` |

執行模式：

```python
RUN_MODE = "new"     # 建立新 run
RUN_MODE = "resume"  # 從既有 output directory 繼續
```

`resume` 模式需正確指定 `RESUME_DIR`。搜尋期間會持續寫入 checkpoint 與 progress files，因此中斷後可從已儲存狀態繼續。

常見 recursive search 輸出包括：

- `search_progress.csv`
- `proposal_history_checkpoint.csv`
- `current_elements_checkpoint.csv`
- `current_validation.json`
- `search_checkpoint.json`
- `final_elements.csv`

所有 generated outputs 預設位於根目錄的 `outputs/`，且不納入 Git。

## Model checkpoint 說明

- `.pt` 檔是已訓練的 model parameters/checkpoint。
- checkpoint 存在時，推論與 recursive design 不需要重新訓練。
- 只有改變 model architecture、training data、training objective，或需要重新估計 weights 時才需重訓。
- recursive design 的結果依賴所選 checkpoint、element models、輸入 tables 與 search configuration。

## Reproducibility notes

- 主要 scripts 使用固定 random seed `777`，但不同 PyTorch、CUDA 或硬體環境仍可能造成小幅差異。
- 每次 run 應保留對應的 `training_config.json`、model metadata 與 output configuration。
- 不要將大型 raw data、generated outputs、secrets 或暫存檔 commit 至 Git。
- Jupyter notebooks 在 commit 前建議清除不必要的 cell outputs，以減少 repository 大小並避免留下本機路徑或大量執行結果。

## Git workflow

修改完成後先檢查差異，再自行 Stage、Commit 與 Push：

```powershell
git status
git diff
git add README.md
git commit -m "Add project README"
git push
```

`git add README.md` 只會 Stage README，不會把其他 untracked files 一起加入 commit。
