# Library release pipeline

Run `01` → `07` in order. Every notebook starts with the same bootstrap cell,
which locates this folder by walking up from the working directory and imports
[`_paths.py`](_paths.py). After that all paths are absolute, so it does not
matter which directory Jupyter was launched from.

```
BG5 + promoter + RE1 + RE2 + BARCODE + BG3
```

Only the flanks are shared. Each promoter enters at its own natural length —
69 nt designed, 50–66 nt native, 75 nt phage — because outside the designed set
we do not know where the real element boundaries sit. The flanks are added in
one place, `06_assemble.ipynb`, and nowhere else.

## Run order

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 01 | `01_recursive_design.ipynb` | `tables/*.pkl`, `weights/*.pt`, `BPM/Params_Con17.pkl` | `outputs/01_final_design.csv` |
| 02 | `02_native_selection.ipynb` | `data/native/Data_S1_20250826.xlsx`, `data/native/*.keg` | `outputs/02_native_promoters.csv` |
| 03 | `03_phage_selection_scan.ipynb` | `data/phage/*.gb` | `outputs/03_phage_promoters.csv` |
| 04 | `04_phage_selection_curated.ipynb` | `data/phage/*.gb`, `data/phage/T5 promoter.xlsx` | `outputs/04_phage_promoters.csv` |
| 05 | `05_barcode_generation.ipynb` | — | `outputs/05_barcodes.csv` |
| 06 | `06_assemble.ipynb` | `outputs/01`–`05` | `outputs/06_whole_sequence.csv` |
| 07 | `07_re_scan.ipynb` | `outputs/06_whole_sequence.csv` | `outputs/07_re_scan_report.csv` |

01–05 are independent of each other and can run in any order, or be skipped if
their output CSV is already present. 06 needs all five. 07 needs 06.

Each of 01–05 ends in a standardising cell that is **self-contained**: it
re-reads that notebook's own result from disk, so it can be re-run without
redoing the expensive work above it.

## Status

Only 01 and 02 have been run end-to-end so far (`outputs/01_final_design.csv`,
`outputs/02_native_promoters.csv` exist). 03–07 are written but not yet
verified against real data — do not treat this pipeline as validated until
`outputs/07_re_scan_report.csv` exists and `qc_pass` has been checked on the
full concatenated table.

## Shared schema

Every candidate table carries these six columns, so 06 can concatenate them:

| Column | Meaning |
|---|---|
| `candidate_id` | `DES-V#####` designed, `NAT-#####` native, `PHG-S#####` phage scan, `PHG-C#####` phage curated |
| `source` | `assembled` / `native` / `phage` |
| `promoter_sequence` | uppercase, no flanks |
| `promoter_length` | |
| `alphabet_valid` | strictly `[ACGT]+` — this is what catches the `*` characters in the native sheets |
| `qc_pass` | `alphabet_valid` and non-empty, plus any source-specific condition |

Source-specific columns ride along and stay joinable on `candidate_id`.

## Changing the construct

`06_assemble.ipynb` cell 3 is the only definition of the flanks:

```python
BG5 = "CACGAGGCCCTTTCGTCTTCACACAGCAGCAGTCAGGTAGGGAAGAGACC"
RE1 = "GTCGAC"      # SalI
RE2 = "TCTAGA"      # XbaI
BG3 = "GGTCTCAAGGCTGCTAACAAAGCCCGAAAGGAAGCTGAGTTGGCTGCTGC"
BARCODE_LEN = 15
```

Change an enzyme there, then re-run 06 and 07. Nothing else needs editing.

06 records the offset of every segment (`promoter_start`, `re1_start`,
`barcode_start`, …) into its output, and 07 uses those offsets to attribute each
restriction site to the segment it came from. A site inside BG5/BG3/RE1/RE2 is
by design; anything in the promoter or barcode, or spanning a junction, is
reported as `sites_unexpected` with a `unexpected_detail` string like
`Eco31I@129(RE2+barcode)`.

## Layout

```
library_release/
├── _paths.py          shared paths, schema helper, RE site table
├── 01..07 *.ipynb
├── data/
│   ├── native/        Data_S1_20250826.xlsx + 12 KEGG .keg
│   └── phage/         17 genome .gb + T5 promoter.xlsx
├── outputs/           all pipeline CSVs land here
└── _duplicates/       byte-identical copies and off-pipeline notebooks,
                       kept only so nothing was deleted; safe to remove
```

Python libraries (`automated_promoter_library_design.py`,
`recursive_corepromoter_design.py`, `anchor_pull_replacement.py`,
`pas_library_qc.py`) and `BPM/` stay in `MS2_Data_PyTorch/scripts/`;
`_paths.py` puts that directory on `sys.path`.

## Notes

- `tables/` is never written to by this pipeline. The older
  `barcodes_15bp_HD3_no_RE_sites.csv` there is byte-identical to the 12 bp
  files and should not be used; 05 regenerates barcodes into `outputs/`.
- `data/phage/*.gb` is matched by the repository-wide `*.gb` gitignore rule, so
  a fresh clone will not have the genomes 03 and 04 need.
- `pas_library_qc.py` is not part of this pipeline. Register checking is done by
  `automated_promoter_library_design.CorePromoterScanner.scan()`, which uses the
  CorePromoter model the design was optimised against.
- That register check runs inside 01, on the 124 nt designed construct, before
  RE1/RE2/barcode/BG3 are attached in 06. Nothing in this pipeline re-checks
  m35/m10 register on the fully flanked `full_sequence`; 07 only scans for
  restriction sites. If barcode/RE placement turns out to perturb the
  -35/-10 register, this pipeline would not currently catch it.
- `data/native/native_promoters_12species.xlsx` and `data/native/kegg_4cat_counts.csv`
  are leftover from copying the audit snapshot in; 02 does not read either one
  (it only reads `Data_S1_20250826.xlsx` and the `*.keg` files). Safe to delete.
