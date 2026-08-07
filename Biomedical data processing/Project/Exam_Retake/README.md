# BDP final project — retake submission

Teodora Taleska, August 2026.

## What to upload

| File | Notes |
|---|---|
| `Part 1/2025_project_part1.ipynb` | Part I code, with outputs |
| `Part 2/2025_project_part2.ipynb` | Part II code, with outputs |
| `report/main.pdf` | The written report (source: `report/main.tex`) |

Do **not** upload `Part 1/ecg_signal.npz`, `Part 2/data/` or `Part 2/cache/` — the first two
were provided with the assignment and the third is regenerable output.

## Running the notebooks

**Part 1** needs `ecg_signal.npz` in the same folder. It runs top to bottom in about
30 seconds and needs nothing but `numpy`, `scipy`, `pywt`, `sklearn`, `pandas`, `matplotlib`.

**Part 2** needs `data/signals/subject_*.pt` and `data/stages/subject_*.pt` in the same
folder — the same layout as the original assignment archive. In this working copy `data` is a
symlink to `../../Part 2/data`; if that does not resolve on your machine, copy or move the real
`data` folder next to the notebook. It additionally needs `torch`, `seaborn` and `joblib`.

Part 2 caches its expensive steps in `Part 2/cache/`:

| Cache file | Step | Cost when recomputed |
|---|---|---|
| `ica.joblib` | FastICA fit | seconds |
| `features.npz` | ICA + filtering + feature extraction, all 20 subjects | ~1 min |
| `cv_results_retake.csv` | the 39-pipeline leave-one-subject-out grid | ~25 min |
| `ic_selection.csv`, `final_model.joblib`, `y_pred.npy` | component audit, deployed model | ~1 min |

With a warm cache the notebook runs in about 40 seconds. **Delete the `cache` folder to
recompute everything from scratch** — the recompute path has been verified to reproduce the
cached cross-validation table exactly. Each cached step also has a `RECOMPUTE_*` flag near it
that can be set to `True` to force a fresh run of that step alone.

The zero-byte files in `cache/` are placeholders left over from the staged run that produced
the cache; they are not read by the notebook and can be deleted.

## Building the report

```
cd report
pdflatex main.tex && pdflatex main.tex
```

`main.tex` uses `siunitx` when it is available (e.g. on Overleaf) and falls back to plain-text
units otherwise, so it compiles on a minimal TeX Live installation as well. All figures come
from `report/figures/`, which both notebooks write on every run, and the `tab_*.tex` files are
LaTeX exports of the tables — the report therefore cannot drift out of sync with the code.

## Changes since the first exam period

Every change is marked in the code with an inline `# RETAKE CHANGE` comment (19 in Part 1,
29 in Part 2) and is listed with its justification in Section 3 of the report.

Headline effect:

- **Part I** — identification accuracy 0.558 → **1.000** (0.773 → 0.996 after dimensionality
  reduction).
- **Part II** — LOSO F1-macro 0.678 → **0.727**; on the held-out subjects, accuracy
  0.744 → **0.780** and F1-macro 0.610 → **0.719**, with every stage improving and N1 rising
  from F1 0.062 to 0.375.
