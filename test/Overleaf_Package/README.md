# Overleaf package — FYP report

Upload this entire folder to [Overleaf](https://www.overleaf.com/) as a **New Project → Upload Project** (zip the folder first if you prefer).

## Contents

| File / folder | Purpose |
|---------------|---------|
| `main.tex` | Main report (set as **Main document** in Overleaf). |
| `references.bib` | IEEE-style bibliography (used with `IEEEtran` + `pdfLaTeX`). |
| `figures/` | All `\includegraphics` assets: analysis plots, `fyp_description.png`, and `ntu_logo.png`. |

## Before submitting

1. **Replace** `figures/ntu_logo.png` with your official NTU logo (the included file is only a grey placeholder).
2. Optionally replace `fyp_description.png` and the notebook-export PNGs if you regenerate figures.
3. In Overleaf: **Menu → Compiler → pdfLaTeX**. Recompile; citations appear after the first full BibTeX pass.

## Syncing from the repo

The canonical LaTeX source in the repository is `test/Final_Project_Report_Overleaf.tex`. After editing that file, regenerate `main.tex` by copying it into this folder and applying the same path changes (`\graphicspath{{figures/}}`, image names, `\bibliography{references}`), or edit `main.tex` here directly.
