# `data/phage/` 資料來源與啟動子選取依據

產生日期：2026-09-21

這份文件回答三件事：每個基因組檔從哪裡來、為什麼選它、裡面取了哪些啟動子以及怎麼取的。
表格由 `library_release/gen_provenance.py` 直接從流程自身的輸出產生，未經人工轉抄；
資料有變動時重跑該腳本即可更新。

---

## 一、基因組檔來源

全部 17 個檔案皆為 NCBI Nucleotide 的 GenBank 格式紀錄。2026-06-10 下載的 13 個檔在下載資料夾
留有紀錄，確認為網頁手動下載；2026-06-08 的 3 個檔未留下下載紀錄。

| `.gb` 檔 | 物種 | 下載日 | 選它的依據 | curated | 掃描候選 |
|---|---|---|---|---:|---:|
| `AY543070.1.gb` | *Escherichia phage T5* | 06-08 | 6/8 單獨下載，依據未留下紀錄 | 11 | 1,718 |
| `EU568876.1.gb` | *Mycobacterium phage BPs* | 06-10 | Dedrick 2023 *CID* 治療用株 | — | 6 |
| `KC787107.1.gb` | *Mycobacterium phage Bo4* | 06-10 | 分枝桿菌噬菌體批次（推定） | — | 3 |
| `NC_000866.4.gb` | *Enterobacteria phage T4* | 06-10 | T 系列批次（推定） | — | 2,836 |
| `NC_001335.gb` | *Mycobacterium phage L5* | 06-08 | 6/8 單獨下載，依據未留下紀錄 | 1 | 32 |
| `NC_001416.gb` | *Enterobacteria phage lambda* | 06-08 | 6/8 單獨下載，依據未留下紀錄 | 2 | 193 |
| `NC_001604.1.gb` | *Enterobacteria phage T7* | 06-10 | T 系列批次（推定） | 6 | 165 |
| `NC_001900.2.gb` | *Mycobacterium virus D29* | 06-10 | Ford, D29 基因組（Zotero 6/9 存入） | 1 | 19 |
| `NC_003387.1.gb` | *Mycobacterium phage TM4* | 06-10 | 分枝桿菌噬菌體批次（推定） | — | 23 |
| `NC_005833.1.gb` | *Escherichia phage T1* | 06-10 | T 系列批次（推定） | — | 326 |
| `NC_005859.1.gb` | *Enterobacteria phage T5* | 06-10 | T 系列批次（推定） | — | 1,718 |
| `NC_022054.2.gb` | *Mycobacterium phage Muddy* | 06-10 | Dedrick 2023 *CID* 治療用株 | — | 44 |
| `NC_023744.1.gb` | *Mycobacterium phage DS6A* | 06-10 | 分枝桿菌噬菌體批次（推定） | — | 41 |
| `NC_024147.1.gb` | *Mycobacterium phage ZoeJ* | 06-10 | Dedrick 2023 *CID* 治療用株 | — | 21 |
| `NC_047864.1.gb` | *Enterobacteria phage T3* | 06-10 | T 系列批次（推定） | — | 150 |
| `NC_054907.1.gb` | *Enterobacteria phage T6* | 06-10 | T 系列批次（推定） | — | 2,852 |
| `NC_054931.1.gb` | *Escherichia phage T2* | 06-10 | T 系列批次（推定） | — | 2,705 |

標「推定」者為同一批次下載、無法指向單一文獻；未標註者有直接對應的文獻。

### 三個下載批次

| 本地時間 | 事件 |
|---|---|
| 6/8 15:11–18:01 | 下載 `NC_001416`(λ)、`AY543070.1`(T5)、`NC_001335`(L5)。這三個後來成為 curated 的來源。 |
| 6/9 17:26–17:29 | 建立 `T5 promoter.xlsx`（檔案內建 metadata：created 2026-06-09 09:26 UTC） |
| 6/9 17:52 | Zotero 存入 Ford, *Genome Structure of Mycobacteriophage D29* |
| 6/10 16:13 | Zotero 存入 Sachdeva 2010 *FEBS J*，M. tuberculosis σ factors |
| **6/10 21:34** | Zotero 存入 Dedrick 2023 *CID*、Yang 2024 *Commun Biol*、González-García 2026 *BRN*（分枝桿菌噬菌體療法） |
| 6/10 21:50–22:15 | 下載 Bo4、D29、TM4、Muddy、DS6A、ZoeJ、BPs 七株分枝桿菌噬菌體 |
| **6/10 22:18** | Zotero 存入 Keith 2024 *PNAS*、Shamsuzzaman 2024 *Front Microbiol*、Nikulin 2023（*E. coli* 噬菌體療法） |
| 6/10 22:47–22:48 | 下載 T1、T2、T3、T4、T5(RefSeq)、T6、T7 七株 |
| 6/11 22:28 | 產出第一版 curated 表，座標與現行版本相同 |

兩次「存文獻 → 下載基因組」間隔 16 分鐘與 29 分鐘。掃描用的基因組是照噬菌體療法文獻實際使用的
株別選的，其中 BPs、Muddy、ZoeJ 是 Dedrick 2023 *CID* 中用於治療的三株。

---

## 二、curated 啟動子（21 條，`04_phage_promoters.csv`）

`+1` 為 1-based 基因組座標，與 NCBI 顯示一致。取窗為 `-60..+15`，共 75 nt。

| 啟動子 | `.gb` | +1 | 股 | 選法 | 依據原文 | 驗證 |
|---|---|---:|:-:|---|---|---|
| `T7_A1` | `NC_001604.1` | 498 | + | GenBank 註解 | `/note="E. coli promoter A1"` @ `498` | ✓ -10 末端在 -8 |
| `T7_A2` | `NC_001604.1` | 626 | + | GenBank 註解 | `/note="E. coli promoter A2"` @ `626` | ✓ -10 末端在 -8 |
| `T7_A3` | `NC_001604.1` | 750 | + | GenBank 註解 | `/note="E. coli promoter A3"` @ `750` | ✓ -10 末端在 -8 |
| `T7_B` | `NC_001604.1` | 1514 | + | GenBank 註解 | `/note="E. coli B promoter"` @ `1514` | ✓ -10 末端在 -8 |
| `T7_C` | `NC_001604.1` | 3113 | + | GenBank 註解 | `/note="E. coli C promoter"` @ `3113` | ✓ -10 末端在 -7 |
| `T7_E[6]` | `NC_001604.1` | 36836 | + | GenBank 註解 | `/note="E. coli promoter E[6]"` @ `36836` | ✓ -10 末端在 -8 |
| `T5_D_E_20` | `AY543070.1` | 4604 | - | GenBank 註解 | `/note="promoter P-D/E 20"` @ `complement(4338..4664)` | ✗ feature 長度 327 而非 75，`location_ok=False`，未進文庫 |
| `T5_H_22` | `AY543070.1` | 4813 | - | GenBank 註解 | `/note="promoter P-H 22"` @ `complement(4799..4873)` | ⚠ +1 偏移 5 nt（-10 末端在 -12） |
| `T5_F_30` | `AY543070.1` | 16724 | - | GenBank 註解 | `/note="promoter P-F 30"` @ `complement(16779..16784)` | ✗ feature 長度 6 而非 75，`location_ok=False`，未進文庫 |
| `T5_D_E_33` | `AY543070.1` | 42241 | - | GenBank 註解 | `/note="promoter P-D/E 33"` @ `complement(42227..42301)` | ⚠ +1 偏移 5 nt（-10 末端在 -12） |
| `T5_H_207` | `AY543070.1` | 45999 | - | GenBank 註解 | `/note="promoter P-H 207"` @ `complement(45985..46059)` | ⚠ +1 偏移 5 nt（-10 末端在 -13） |
| `T5_N_25` | `AY543070.1` | 63103 | + | GenBank 註解 | `/note="promoter P-N 25"` @ `63043..63117` | ⚠ +1 偏移 5 nt（-10 末端在 -12） |
| `T5_N_26` | `AY543070.1` | 64128 | + | GenBank 註解 | `/note="promoter P-N 26"` @ `64068..64142` | ⚠ +1 偏移 5 nt（-10 末端在 -12） |
| `T5_K_28b` | `AY543070.1` | 66426 | + | GenBank 註解 | `/note="promoter P-K 28b"` @ `66366..66440` | ⚠ +1 偏移 5 nt（-10 末端在 -12） |
| `T5_K_28a` | `AY543070.1` | 66519 | + | GenBank 註解 | `/note="promoter P-K 28a"` @ `66459..66533` | ⚠ +1 偏移 5 nt（-10 末端在 -12） |
| `T5_G_25` | `AY543070.1` | 92027 | - | GenBank 註解 | `/note="promoter P-G 25"` @ `complement(92013..92087)` | ⚠ +1 偏移 5 nt（-10 末端在 -12） |
| `T5_J_5` | `AY543070.1` | 99924 | - | GenBank 註解 | `/note="promoter P-J 5"` @ `complement(99910..99984)` | ⚠ +1 偏移 5 nt（-10 末端在 -12） |
| `lambda_PR` | `NC_001416.1` | 38023 | + | 文獻座標 | Hawley&McClure 1983　**待查證** | ✓ -10 末端在 -7 |
| `lambda_PL` | `NC_001416.1` | 34560 | - | 文獻座標 | Hawley&McClure 1983　**待查證** | ✗ 取到 `complement()` 左端，+1 應為 **35582** |
| `L5_Pleft` | `NC_001335.1` | 51672 | - | 文獻座標 | Brown 1997 EMBO; Dedrick 2017 BMC Microbiol　**待查證** | ✓ -10 末端在 -7 |
| `D29_Pleft` | `NC_001900.2` | 48503 | - | 文獻座標 | Brown 1997 EMBO; Dedrick 2017 BMC Microbiol　**待查證** | ✓ -10 末端在 -7 |

### 三種選法的意思

- **GenBank 註解**：`.gb` 檔裡本來就有 `regulatory_class="promoter"` 的標註，直接讀它的位置。
  要複查的話，打開該 `.gb` 搜尋「依據原文」欄的字串即可，不必透過本流程。
- **文獻座標**：`.gb` 沒有標註，由人工依文獻填入 `+1`。`NC_001416`(λ) 完全沒有啟動子類標註，
  `NC_001335`(L5) 與 `NC_001900.2`(D29) 只有 CDS/gene/tRNA。
- 本流程**沒有使用 BLAST 或任何序列比對**。

### 驗證欄怎麼算的

用 `04_phage_promoters.csv` 裡現成的 `bpm_core_end_in_full_1based`，量 -10 六聯體末端到 `+1` 的
距離（discriminator 長度）。σ70 啟動子此距離通常為 5–8 nt。座標無誤的 11 條全部落在 6–7 nt。

### T7 六條的選取範圍

`NC_001604.1` 中標為 `E. coli promoter` 的標註共 7 個：A0(leftward)、A1、A2、A3、B、C、E[6]。
取其中 6 個，未取註明 leftward 的 A0。其餘 `T7 promoter phiXX` 標註為 T7 RNAP 專用，
不受宿主 RNAP 辨識，故排除。

### T5 十一條的選取範圍

`AY543070.1` 中帶 `/note` 的啟動子標註共 26 個。取的是命名為「P-<字母> <數字>」的 11 個；
未取 P10a、P11、P12、P13a/b、P14a/b、P15a/b/c、P 16-17、P31A/B、P-31A/B 等純數字命名者。
`T5 promoter.xlsx`（6/9 建立，只有 name 與 location 兩欄）即為此 11 條的抄錄。

其中 `P-N 25` 即 pQE 系列載體使用的 T5 啟動子：本流程切出的序列含
`AAATCATAAAAAATTTATTTGCTT`，其 17 bp spacer `TCAGGAAAATTTTTCTG` 正是 pQE 中被置換為
lac operator 的那一段。

---

## 三、掃描啟動子（`03_phage_promoters.csv`）

這批沒有已知 TSS，由 `BPM` core promoter 模型計算取得，不涉及文獻或標註。

| 設定 | 值 |
|---|---|
| 掃描對象 | `data/phage/*.gb` 全部 17 個（`ACCESSIONS = None`） |
| 基因組總長 | 約 1.32 Mbp |
| 列舉方式 | 兩股 × spacer 長度 15/16/17/18/19，共約 13.2 M 個 window |
| 保留條件 | `MIN_LOGEXP = 2.0`（於每個基因組的掃描迴圈內即套用） |
| 輸出 | 12,852 條候選，取窗同為 `-60..+15` |

每個基因組的候選數見第一節表格最後一欄。

---

## 四、已知問題

**1. `lambda_PL` 座標取錯端。**
`NC_001416.gb` 中 `mRNA complement(34560..35582) /product="mRNA-pl"`，負股轉錄起點為右端
35582，填入的 34560 是該轉錄本的 3' 末端。三項證據一致指向 35582：10 個 `mRNA-pl` 系列標註
全部以 35582 為起點；典型 PL 序列 `GTGTTGACATAAATACCACTGGCGGTGATACTGAG` 位於負股 35586
附近；`operator-l1/l2/l3` 標註在 35591–35651，緊鄰 35582 上游。對照 `lambda_PR` 取自
`mRNA 38023..40624`，正股取左端故正確。**此條目前 `qc_pass=True`，已進入文庫。**

**2. T5 九條 `+1` 系統性偏移 5 nt。**
長度正常的 9 條，-10 末端全部落在 -13 而非 -7/-8。推測 `AY543070.1` 的啟動子標註並非
`-60..+15` 而接近 `-65..+10`，需修正 04 的 `T5_FULL_WINDOW` 假設。

**3. T5 基因組重複收錄。**
`AY543070.1` 與 `NC_005859.1` 序列逐鹼基完全相同（皆 121,750 bp），兩者皆被掃描，
各產生 1,718 條候選。12,860 條 phage 列中有 4,344 條序列重複。

**4. 出處欄位未進入最終表。**
04 的 `CANDIDATES` 每列皆有 `source` 欄記錄出處，但標準化步驟
`standardize(curated, source="phage", ...)` 將其覆寫為來源類別，故 `04_phage_promoters.csv`
與 `06_whole_sequence.csv` 均看不到出處。原始內容保留於 `04_phage_curated_raw.csv`。

---

## 五、待查證

以下引用存在於 04 的 `source` 欄位，但 Zotero（308 筆）與下載資料夾中均無對應文獻，無法確認
是否實際查閱過。座標本身的可信度見第二節「驗證」欄，與這些引用是否成立無關。

| 對象 | 現有引用 | 缺什麼 | 建議補法 |
|---|---|---|---|
| `T7_A1`–`T7_E[6]` | `Dunn & Studier 1983` | 庫內無此文獻 | 座標已由 `NC_001604.1` 標註驗證；引用可改用該紀錄內含的 Prosen & Cech 1985 *Biochemistry*（PMID 3922411，對應 E promoter） |
| T5 十一條 | `T5 promoter.xlsx` + `AY543070.1` 標註 | 標註本身的原始出處 | `AY543070.1` 含 Kaliman 等人關於 T5 啟動子結構的文獻；另需確認標註的視窗定義以解決問題 2 |
| `lambda_PR`、`lambda_PL` | `Hawley & McClure 1983` | 庫內無此文獻 | `NC_001416` 紀錄內含 Dahlberg & Blattner 1975 *NAR*（PMID 1178525）、Kleid 等 1975 *JBC*（PMID 167018）、Walz 等 1976 *Nature*（PMID 958438）、Horn & Wells 1981 *JBC*（PMID 6257696） |
| `D29_Pleft` | `Brown 1997 EMBO; Dedrick 2017 BMC Microbiol` | 兩筆皆查無；Zotero 有 Ford, D29 基因組（缺年份欄位） | 補齊 Ford 該文書目；`Dedrick 2017 BMC Microbiol` 疑為 Dedrick 2023 *CID*（doi:10.1093/cid/ciac453）之誤記 |
| `L5_Pleft` | `Brown 1997 EMBO; Dedrick 2017 BMC Microbiol` | 同上，且 `.gb` 無標註、Zotero 無 L5 相關文獻 | **21 條中支撐最弱的一條**，建議查證後再決定是否保留 |

