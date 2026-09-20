# weights/

這個目錄的檔案**有進 Git**（`.gitignore` 對 `*.pt` 的排除規則特別為它開了例外）。

## 整體模型

| 檔案 | 誰載入 | 說明 |
|---|---|---|
| `weights_CorePromoter_clean.pt` | `01_recursive_design.ipynb`、保守度分析、register 診斷、能量軸比較 | 凍結的 baseline。`Model_CorePromoter_clean.ipynb` **不會寫出這個檔**，它是手動存的；要更新得自己存檔並確認下游是否要重跑 |
| `weights_CorePromoter_clean_0711.pt` | 無 | 舊版 baseline，留作對照 |

`weights_CorePromoter_clean_{history.csv, metadata.json, training_counts.csv}` 是對應的訓練紀錄。

## 元件模型

六個檔案都由 `ElementModelBundle._load_all()`（`automated_promoter_library_design.py`）載入，載入方式是 `strict=True`，**缺任何一個都會讓 01 在建立 bundle 那一格直接 `FileNotFoundError`**。

| 檔案 | 由誰訓練 | 是否參與打分 |
|---|---|---|
| `weights_UP.pt` | `Model_UP.ipynb` | 是 |
| `weights_Sp17.pt` | `Model_Sp17.ipynb` | 是 |
| `weights_Dis.pt` | `Model_Dis.ipynb` | 是 |
| `weights_ITS.pt` | `Model_ITS.ipynb` | 是 |
| `weights_Sp16.pt` | `Model_Sp16.ipynb` | **否，但不能刪** |
| `weights_Sp18.pt` | `Model_Sp18.ipynb` | **否，但不能刪** |

Sp16 與 Sp18 不參與打分的原因：設計流程的 spacer 序列池唯一來源是 `SOURCE_SPECS["spacer"] = ("SL17.pkl", None)`，而 `ELEMENT_LENGTHS["spacer"] = 17`，所以池子裡全部是 17 nt，`score()` 依長度分派時永遠只會用到 Sp17。衍生的 `Spacer_v3 = Spacer_v2[:-2] + "TG"` 也還是 17 nt。但 `_load_all` 的迴圈是 `for length in (16, 17, 18)`，三個都會載。

另一個相關的細節：快取簽章只記 `weights_Sp17.pt`，所以重訓 Sp16 或 Sp18 **不會**讓 `outputs/energy_bin_cache/` 失效。今天沒有影響，但若哪天把 `ELEMENT_LENGTHS["spacer"]` 改成 16 或 18，要記得手動清快取。

## -35 / -10

`weights_minus35_unused.pt`、`weights_minus10_unused.pt` **沒有任何程式載入**。設計流程的 -35/-10 一律用 BPM：`ElementModelBundle.score()` 回傳 `-BPM dG`，`build_scored_pools` 也把這兩個元件的權重簽章指向 `BPM/Params_Con17.pkl`。

保留它們的理由是它們與 BPM 排序嚴重不一致（Spearman ρ = 0.37 / −0.22），這個分歧本身值得能重現。已知狀況：

- 它們的結構是 `ElementEnergyModel` 的 state dict（`conv1.weight (1, 4, 6)` 加 `param_max` / `param_min` / `param_e0`），也就是**類神經網路訓練出來的，不是從 BPM 提取的**。
- 產生它們的程式碼**不在這個 repo 裡**。`Model_PL.ipynb` 沒有任何 `torch.save`，git 歷史顯示它們是在最初的大 commit 一次進來的。

## 匯出權重的注意事項

各元件 notebook 結尾的匯出格預設 `OVERWRITE = False`，只會比對差異、不會寫檔。目前六個 `.pt` 的檔案時間都是 2026-07-23，而且沒有任何 `weights_*_metadata.json`，代表**這條匯出路徑從來沒有真的執行過寫入**。第一次把 `OVERWRITE` 改成 `True` 的人請先確認：

- 只有 `conv1.weight`、`param_max`、`param_min`、`param_e0` 四個鍵可以存進去。notebook 模型多出來的 `conv1.bias`（恆為 0）和 `fc1`–`fc3`（forward 沒用到）必須濾掉，整包存下去會讓 `strict=True` 的載入直接失敗。
- 存完後那格會用 `ElementEnergyModel` 回讀一次並比對能量，差異應小於 `1e-5`。
