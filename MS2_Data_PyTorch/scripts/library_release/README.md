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
| 02 | `02_native_selection.ipynb` | `data/native/Data_S1_20250826.xlsx`, `data/native/*.keg` | `data/native/native_promoters_12species.xlsx`, `data/native/kegg_4cat_counts.csv`, `outputs/02_native_promoters.csv` |
| 03 | `03_phage_selection_scan.ipynb` | `data/phage/*.gb` | `outputs/03_phage_scan_candidates_raw.csv`, `outputs/03_phage_promoters.csv` |
| 04 | `04_phage_selection_curated.ipynb` | `data/phage/*.gb`, `data/phage/T5 promoter.xlsx` | `outputs/04_phage_curated_raw.csv`, `outputs/04_phage_promoters.csv` |
| 05 | `05_barcode_generation.ipynb` | — | `outputs/05_barcodes.csv` |
| 06 | `06_assemble.ipynb` | `outputs/01`–`05` | `outputs/06_whole_sequence.csv` |
| 07 | `07_re_scan.ipynb` | `outputs/06_whole_sequence.csv` | `outputs/07_re_scan_report.csv` |

01–05 are independent of each other and can run in any order, or be skipped if
their output CSV is already present. 06 needs all five. 07 needs 06.

Each of 01–05 ends in a standardising cell that is **self-contained**: it
re-reads that notebook's own result from disk, so it can be re-run without
redoing the expensive work above it.

## Status

All seven notebooks have now been run end-to-end and every output in the table
above exists, `outputs/07_re_scan_report.csv` included.

01 is the exception worth knowing about: its search (Batch 6) was not re-run.
Batch 8 re-exports from the design run recorded in the `design_run` column of
`outputs/01_final_design.csv`. It now prefers this notebook's own `OUT_DIR` and
only falls back to `latest_design_run()` — printing a warning — when the current
run produced no `final_scan_15625.csv`. Before that fix an interrupted search
would leave a newer directory with no `final_scan`, and Batch 8 silently exported
an older, unrelated design.

What is *not* validated is the biology: `07_re_scan_report.csv` shows a large
number of restriction sites inside the promoters themselves (see
[Restriction sites](#restriction-sites) below). Those are a property of the
sequences, not a pipeline bug, and they still need a decision.

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

RE2 and the first nucleotides of BG3 are also duplicated in 05, as
`CONTEXT_PREFIX` / `CONTEXT_SUFFIX`, because a barcode has to be screened in the
context it will actually sit in — see the next section. 06 reads the context 05
recorded in `05_barcodes.csv` and raises if it no longer matches the flanks
defined here, so the two cannot drift apart silently.

## Restriction sites

05 screens each barcode inside the context it will actually sit in,
`RE1 + RE2 | barcode | BG3`, and counts only hits that **overlap the barcode** —
the flanks carry sites by design (`CONTEXT_PREFIX` ends in XbaI because RE2 *is*
XbaI), so a plain substring test on the flanked string rejects every candidate.
It is the same expected/unexpected split 07 applies to the assembled construct,
and the two agree row for row.

Screening the bare 15-mer instead let 1,546 sites through at the two barcode
junctions, across 1,532 constructs — 125 of them Eco31I (BsaI), the Golden Gate
enzyme built into BG5/BG3, which would have broken assembly. After the change
`07_re_scan_report.csv` reports **zero** sites touching a barcode, and the clean
rate goes from 58.1% to 61.0% (20,727 / 33,999).

What remains is entirely promoter-related: 16,592 sites inside promoters plus 384
straddling a promoter boundary. None of the 19 enzymes in `_paths.RE_SITES` is
free of sites across the whole library, so swapping RE1/RE2 for a different pair
does not help. Native and phage promoters carry whatever the real sequence
carries and cannot be edited; the designed set is the one place this is
controllable, and its element pools are currently not RE-filtered — which is why
`assembled` has the *worst* clean rate of the three sources (40.8%, against 82.6%
native and 76.2% phage). Read `sites_unexpected` and `unexpected_detail` in
`outputs/07_re_scan_report.csv` per source before ordering.

## Layout

```
library_release/
├── _paths.py          shared paths, schema helper, RE site table
├── 01..07 *.ipynb
├── data/
│   ├── native/        Data_S1_20250826.xlsx + 12 KEGG .keg
│   └── phage/         17 genome .gb + T5 promoter.xlsx
├── outputs/           all pipeline CSVs land here
└── (_duplicates/ was moved to _archive_20260918/ on 2026-09-18)
```

Python libraries (`automated_promoter_library_design.py`,
`recursive_corepromoter_design.py`) and `BPM/` stay in
`MS2_Data_PyTorch/scripts/`;
`_paths.py` puts that directory on `sys.path`.

## Notes

- `tables/` is never written to by this pipeline. Its four barcode CSVs were
  moved to `_archive_20260918/` on 2026-09-18; 05 regenerates barcodes into
  `outputs/`.
- `data/phage/*.gb` is matched by the repository-wide `*.gb` gitignore rule, so
  a fresh clone will not have the genomes 03 and 04 need.
- Register checking is done by
  `automated_promoter_library_design.CorePromoterScanner.scan()`, which uses the
  CorePromoter model the design was optimised against.
- That register check runs inside 01, on the 124 nt designed construct, before
  RE1/RE2/barcode/BG3 are attached in 06. Nothing in this pipeline re-checks
  m35/m10 register on the fully flanked `full_sequence`; 07 only scans for
  restriction sites. If barcode/RE placement turns out to perturb the
  -35/-10 register, this pipeline would not currently catch it.
- `data/native/native_promoters_12species.xlsx` and `data/native/kegg_4cat_counts.csv`
  are **written by 02, not leftovers**. 02 cell 6 writes the counts CSV, cell 12
  writes the workbook, and the standardising cell reads that workbook back to
  produce `outputs/02_native_promoters.csv`. Deleting the workbook breaks 02's
  last cell. (An earlier revision of this file said they were safe to delete.
  They are not.)
- 07 writes one file, `outputs/07_re_scan_report.csv`. The older
  `outputs/re_scan_detail/` directory came from a whole-file scanner that has
  been removed — it counted the by-design BG5/BG3/RE1/RE2 sites into its `clean`
  column, so that column was always 0%. Those files are stale and can go.
- Re-running 05 replaces every barcode, and 06 pairs barcodes to candidates by
  position in `candidate_id` order, so the whole barcode↔candidate mapping
  changes with it. Do not re-run 05 once anything has been ordered or sequenced
  against the current mapping.
- Two entries in `data/phage/T5 promoter.xlsx` have feature lengths that are not
  75 bp (`P-D/E 20` is 327, `P-F 30` is 6), so the +1 inferred from their
  upstream edge cannot be trusted. 04 marks them `location_ok=False`, which
  forces `qc_pass=False`, and 06 leaves them out. To use them, confirm the real
  +1 against the source literature first.
- 03 applies `MIN_LOGEXP` inside the per-genome loop. Scanning all 17 genomes on
  both strands at five spacer lengths is 13.2 M windows; keeping them all as rows
  before filtering cost ~8 GB and the concat doubled it. Do not move the filter
  back out of `scan_record()`.
