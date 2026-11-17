"""Analysis code behind notebooks/analysis.ipynb (pre-registered tests H1=f_14, H2=f_39)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .helpers import clean_events, load_pack

FEATURES = [f"f_{i:02d}" for i in range(1, 46)]
LOCKED = {"H1": "f_14", "H2": "f_39"}
MIN_EFFECT = 0.05
ALPHA = 0.025  # Bonferroni over the 2 locked tests


def prepare(data_dir) -> dict:
    """Load a pack, clean events, drop the planted canary issuer."""
    pack = load_pack(data_dir)
    raw_n = len(pack["events"])
    ev = clean_events(pack["events"])
    ev = ev[~ev["issuer_id"].str.contains("canary", na=False)]
    return {**pack, "events_clean": ev, "dedup_removed": raw_n - len(ev)}


def pit_sample(ev: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Events published inside their event month, joined to that month's outcome."""
    ok = ev[ev["published_at"].dt.strftime("%Y-%m") == ev["month"]]
    return ok.merge(outcomes, on=["issuer_id", "month"], how="inner")


def clip_points(ev: pd.DataFrame, features) -> dict:
    return {f: ev[f].quantile([0.01, 0.99]).values for f in features}


def lpm(df: pd.DataFrame, f: str, cut=None) -> dict:
    """OLS of deterioration_6m on the standardized feature, issuer-clustered SEs."""
    s = df[[f, "deterioration_6m", "issuer_id"]].dropna()
    x = s[f].clip(*cut[f]) if cut else s[f]
    x = (x - x.mean()) / x.std()
    r = sm.OLS(s["deterioration_6m"].astype(float), sm.add_constant(x)).fit(
        cov_type="cluster", cov_kwds={"groups": s["issuer_id"]}
    )
    b, se, p2 = float(r.params.iloc[1]), float(r.bse.iloc[1]), float(r.pvalues.iloc[1])
    return dict(n=int(len(s)), estimate=b, ci=[b - 1.96 * se, b + 1.96 * se],
                p_one_sided=p2 / 2 if b > 0 else 1 - p2 / 2, p_two_sided=p2)


def verdict(res: dict) -> str:
    return "keep" if res["estimate"] >= MIN_EFFECT and res["p_one_sided"] < ALPHA else "kill"


def with_exits_as_deteriorated(ev, outcomes, exits, how="default_or_distressed"):
    """Survivorship sensitivity: events of issuers with no outcome rows get y=1 if they defaulted."""
    bad = set(exits.loc[exits["exit_reason"].isin(["default", "distressed_exchange"]), "issuer_id"])
    ok = ev[ev["published_at"].dt.strftime("%Y-%m") == ev["month"]]
    missing = ok[~ok["issuer_id"].isin(outcomes["issuer_id"].unique()) & ok["issuer_id"].isin(bad)].copy()
    missing["deterioration_6m"] = 1
    return pd.concat([pit_sample(ev, outcomes), missing], ignore_index=True)


def lpm_adjusted(df: pd.DataFrame, issuers: pd.DataFrame, f: str, cut, controls=("month", "sector", "size_bucket", "source_class")) -> dict:
    """Same model as `lpm` plus fixed effects for the given controls (robustness only)."""
    t = df.merge(issuers[["issuer_id", "sector", "size_bucket"]], on="issuer_id", how="left")
    t = t.dropna(subset=[f, "deterioration_6m"])
    x = t[f].clip(*cut[f])
    x = (x - x.mean()) / x.std()
    X = pd.concat([pd.DataFrame({"x": x.values}),
                   pd.get_dummies(t[list(controls)], drop_first=True).astype(float).reset_index(drop=True)], axis=1)
    r = sm.OLS(t["deterioration_6m"].astype(float).values, sm.add_constant(X).values).fit(
        cov_type="cluster", cov_kwds={"groups": t["issuer_id"].values})
    b, se, p2 = float(r.params[1]), float(r.bse[1]), float(r.pvalues[1])
    return dict(n=int(len(t)), estimate=b, ci=[b - 1.96 * se, b + 1.96 * se],
                p_one_sided=p2 / 2 if b > 0 else 1 - p2 / 2, p_two_sided=p2)
