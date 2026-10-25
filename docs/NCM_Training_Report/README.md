# NCM training report

This folder contains the R/Quarto source and rendered PDF for the TBS NCM training report.

- `ncm_training.qmd` — Quarto source (R/knitr, ggplot2, Mermaid)
- `ncm_training.pdf` — rendered A4 PDF
- `assets/overview.png` — NCM overview figure
- `data/` — small snapshots of the training summaries, metrics, and reports used by the R chunks
- `ncm_training_files/` — generated figure assets from the last render (safe to delete; Quarto recreates them)

## Render

From this directory:

```powershell
quarto render ncm_training.qmd --to pdf
```

The document uses `xelatex` via the installed MiKTeX distribution. If Quarto cannot find the TeX installation, install TinyTeX with `quarto install tinytex` or set `pdf-engine` to another available LaTeX engine.

## Data sources

The base counts, persona counts, league results, self-play round, and representative training curves were copied from:

- `C:\AIs\NCM\tbs_ncm\data\`
- `C:\AIs\NCM\artifacts\tbs\runs\`
- `C:\AIs\NCM\artifacts\tbs\personas\`
- `C:\AIs\NCM\artifacts\tbs\vs_ncm\run1\`

The base 4-epoch generator run predates the monitoring dashboard; its exact-match figures are quoted from `tbs_ncm/README.md`. The curves in the report come from a later monitored persona fine-tune and are labeled as such.
