# Question

Can a vendor-derived signal predict six-month credit deterioration (`deterioration_6m`) if the joins only use information that had actually been published at the time?

# What I tested and why

I screened all 45 features on discovery data only. Roughly 42% of events are published 60 to 130 days after they happen (the revised fields), and a join on event month quietly uses those future revisions, so I only kept events published inside their own event month.

I locked two hypotheses before the holdout was released:

- H1, `f_14` (liquidity). It was the only feature that clearly stood out, at about +33 percentage points per SD.
- H2, `f_39` (operational). Much weaker, about +5.2 points per SD, and borderline.

Both tests use a linear probability model on the standardized feature with issuer-clustered errors, and need an effect of at least 0.05 with a one-sided p below 0.025. The mechanism labels are stories I chose afterwards, since the features are anonymised.

# What survived

H1 held up on the holdout (new events, no overlap with discovery). The slope is +0.318 per SD (95% CI 0.300 to 0.337). It barely changes if I drop exited issuers, skip the clipping, count defaulted issuers as deteriorated, or add controls (0.320 with month, sector, size and source).

H2 passed the locked test but I don't trust it. The slope is +0.067 (CI 0.039 to 0.096), yet once I control for sector it drops to +0.031 (CI -0.006 to 0.068), which is under the 0.05 bar. I'm calling it inconclusive.

# What I killed and why

`f_08` looked big (+19 points) under a naive join and disappeared on the point-in-time sample: leakage from late revisions. `f_05` and `f_06` were around +4.6 to +4.8 points in discovery but can't be told apart from noise once you account for searching 45 features, so I didn't register them.

# How I checked I was not fooling myself

- I locked the hypotheses before downloading the holdout, and every holdout row in the log is timestamped after that.
- A permutation test over all 45 features gave `f_14` a family-wise p under 0.005, but `f_39` only 0.27.
- A timing placebo shows `f_14` only predicts the same-month outcome.
- I rebuilt the numbers separately in raw SQL and got the same results.
- Sector controls shrink `f_39` in discovery too (0.052 to 0.024). I hadn't tried that before registering.
- I fixed the inherited helpers (look-ahead shifts, holdout pooling in `winsorize`, a join that ignored publication time).

# Limitations

`f_14` separates the outcome almost perfectly: the top three deciles all deteriorated. Real credit signals aren't that clean, so it may be synthetic or label-linked. The holdout rules out noise, but not leakage.

Events also cluster where deterioration happens (44% versus 13% without an event), so the results only describe issuer-months with an event. Defaulted issuers have no outcome rows, so deterioration is probably understated.

# How to reproduce

Make a virtual environment (`python3 -m venv .venv`, then `source .venv/bin/activate`) and run `make setup`. Put the discovery pack in `data/`, point `TAKEHOME_HOLDOUT_DIR` at the holdout folder, and run `jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb`. Then run `make test` and `make check`. The notebook writes `submit/results.json`. I used Python 3.13.
