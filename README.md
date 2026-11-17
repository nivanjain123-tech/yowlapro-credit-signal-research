# Point-in-time credit signal research

Research code for a quantitative research take-home: can a vendor-derived signal predict six-month credit deterioration when every join uses only information that had been published at the time?

## What is in here
- `src/takehome/` - fixed helper functions (`helpers.py`), analysis code (`analysis.py`) and the structure validator (`validation.py`).
- `notebooks/analysis.ipynb` - executed notebook: data checks, hypothesis pre-registration, discovery screen, holdout tests, robustness.
- `queries/` - three DuckDB SQL queries.
- `tests/` - pytest regression tests for the look-ahead bugs fixed in `helpers.py`.
- `submit/` - report, experiment log, results and AI-usage note.
- `scripts_work/` - exploratory and independent-rebuild scripts used along the way.

## Not included
The datasets (discovery and holdout) are not part of this repository. Point `TAKEHOME_DATA_DIR` and `TAKEHOME_HOLDOUT_DIR` at local copies to rerun. Some scripts in `scripts_work/` contain absolute local paths.

## Reproduce
```bash
python3 -m venv .venv && source .venv/bin/activate
make setup
make test
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
make check
```
Python 3.13 with the pinned versions in `requirements.txt`.
