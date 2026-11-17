import nbformat as nbf
nb=nbf.v4.new_notebook(); C=[]
md=lambda s:C.append(nbf.v4.new_markdown_cell(s)); code=lambda s:C.append(nbf.v4.new_code_cell(s))
md("# Setup and data loading\n\nResearch question: does a vendor-derived signal predict six-month credit deterioration after careful point-in-time joins?\n\nDiscovery data are read from `data/` (or `TAKEHOME_DATA_DIR`). The holdout is only loaded in the *Holdout tests* section, from `TAKEHOME_HOLDOUT_DIR` (default `../holdout`).")
code('''import os, sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from takehome.analysis import *
from takehome.helpers import load_pack
D = prepare(os.environ.get("TAKEHOME_DATA_DIR", ROOT / "data"))
ev, out, iss, exits = D["events_clean"], D["outcomes"], D["issuers"], D["exits"]
cut = clip_points(ev, FEATURES)
dv = pit_sample(ev, out)
print("discovery: events", len(D["events"]), "-> after dedup/canary removal", len(ev), "| outcomes", out.shape, "| issuers", len(iss), "| exits", len(exits))''')
md("# Observations\n\nData checks on the discovery pack (all counts are computed below).")
code('''raw = D["events"]
feats = [c for c in raw if c.startswith("f_")]
print("rows removed by dedup/canary:", D["dedup_removed"], "| duplicate event_ids in raw:", int(raw.event_id.duplicated().sum()))
print("share of feature cells == -999 (raw):", round(float((raw[feats] == -999).mean().mean()), 4), "-> set to NaN")
print("event_tz values:", sorted(raw.event_tz.unique()), "-> event_time converted to UTC")
lag = (ev.published_at - ev.event_time_utc).dt.days
print("publication lag (days) buckets:", lag.groupby(pd.cut(lag, [-1, 10, 30, 60, 130])).size().to_dict())
in_month = ev.published_at.dt.strftime("%Y-%m") == ev.month
print(f"events published inside their event month: {in_month.mean():.1%}; the rest arrive 60-130 days late (revisions) and cannot be used at event month")
print("issuers with events but no outcome rows:", len(set(ev.issuer_id) - set(out.issuer_id)), "| of which exited:", len((set(ev.issuer_id) - set(out.issuer_id)) & set(exits.issuer_id)))
print("deterioration rate: all issuer-months", round(out.deterioration_6m.mean(), 3), "| issuer-months with a usable event", round(dv.deterioration_6m.mean(), 3))
print("point-in-time-valid discovery sample:", len(dv), "events,", dv.issuer_id.nunique(), "issuers")''')
md("# Pre-registered hypotheses\n\nLocked in the portal before the holdout was released (both confirmatory, direction: higher predictor means more deterioration; minimum effect 0.05 probability per 1 SD; pass needs one-sided p < 0.025, Bonferroni for 2 tests).\n\n| id | predictor | mechanism | idea |\n|---|---|---|---|\n| H1 | `f_14` | liquidity | higher f_14 = tighter funding or cash strain, so deterioration within six months is likelier |\n| H2 | `f_39` | operational | higher f_39 = weaker operating performance, so credit quality erodes over six months |\n\n**Test plan (both):** holdout events published within their event month, joined to `deterioration_6m` on issuer-month; linear probability model on the standardized feature (clipped at discovery 1st/99th percentiles), issuer-clustered SEs.\n\n**Falsifier (both):** holdout slope per 1 SD below 0.05, or not positive, or one-sided clustered p above 0.025.\n\nThe mechanism labels are post-hoc stories: the features are anonymised, so they cannot be verified from the data.")
md("# Cheap first-pass tests\n\nDiscovery only. Screen of all 45 features on the point-in-time-valid sample (OLS slope per 1 SD, issuer-clustered, Benjamini-Hochberg across 45), and a check of how the join changes the answer.")
code('''rows = []
for f in FEATURES:
    r = lpm(dv, f, cut); rows.append((f, r["n"], r["estimate"] * 100, r["p_two_sided"]))
scr = pd.DataFrame(rows, columns=["feature", "n", "pp_per_sd", "p"]).set_index("feature")
scr["q_bh"] = sm.stats.multipletests(scr.p, method="fdr_bh")[1]
print(scr.reindex(scr.pp_per_sd.abs().sort_values(ascending=False).index).head(8).round(4))
# join sensitivity: naive event-month join (ignores publication time) vs point-in-time
naive = ev.merge(out, on=["issuer_id", "month"])
print("\\nslope (pp per SD) naive join vs point-in-time:", {f: (round(lpm(naive, f, cut)["estimate"] * 100, 1), round(lpm(dv, f, cut)["estimate"] * 100, 1)) for f in ["f_08", "f_14", "f_39"]})''')
md("# Holdout tests\n\nRun once, exactly as pre-registered. Holdout events are new (no event_id or issuer-month overlap with discovery).")
code('''HOLD = Path(os.environ.get("TAKEHOME_HOLDOUT_DIR", ROOT.parent / "holdout"))
H = prepare(HOLD)
hev = H["events_clean"]
hv = pit_sample(hev, H["outcomes"])
cut2 = {f: cut[f] for f in LOCKED.values()}
print("holdout: events", len(H["events"]), "-> clean", len(hev), "| published in event month and matched:", len(hv), "| issuers", hv.issuer_id.nunique(), "| deterioration rate", round(hv.deterioration_6m.mean(), 3))
print("overlapping issuer-months with discovery sample:", len(set(zip(dv.issuer_id, dv.month)) & set(zip(hv.issuer_id, hv.month))))
HOLDOUT = {}
for h, f in LOCKED.items():
    r = lpm(hv, f, cut2); r["decision"] = verdict(r); HOLDOUT[h] = r
    print(h, f, f"slope/SD={r['estimate']:.4f}  95% CI=[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]  one-sided p={r['p_one_sided']:.3g}  n={r['n']}  ->", r["decision"])
print("\\nRobustness (same plan, holdout):")
exited = set(H["exits"].issuer_id); hx = with_exits_as_deteriorated(hev, H["outcomes"], H["exits"])
ISS = H["issuers"][["issuer_id", "size_bucket"]]
for h, f in LOCKED.items():
    uniq = hv.groupby(["issuer_id", "month"], as_index=False)[[f, "deterioration_6m"]].mean()
    print(h, f, "| no clipping", round(lpm(hv, f)["estimate"], 4), "| one row/issuer-month", round(lpm(uniq, f, cut2)["estimate"], 4),
          "| drop exited", round(lpm(hv[~hv.issuer_id.isin(exited)], f, cut2)["estimate"], 4),
          "| exited defaulters counted as deteriorated", round(lpm(hx, f, cut2)["estimate"], 4))
    print("   halves", {("early" if k else "late"): round(lpm(g, f, cut2)["estimate"], 3) for k, g in hv.groupby(hv.month < "2022-07")},
          "| size", {k: round(lpm(g, f, cut2)["estimate"], 3) for k, g in hv.merge(ISS, on="issuer_id").groupby("size_bucket")},
          "| source", {k: round(lpm(g, f, cut2)["estimate"], 3) for k, g in hv.groupby("source_class")})''')
code('''print("Adjusted for controls (robustness, holdout): slope per 1 SD [95% CI] with fixed effects")
CTRL = {"none": (), "month": ("month",), "sector": ("sector",), "size": ("size_bucket",), "source": ("source_class",), "all four": ("month", "sector", "size_bucket", "source_class")}
ADJ = {}
for h, f in LOCKED.items():
    row = {}
    for name, c in CTRL.items():
        r = lpm_adjusted(hv, H["issuers"], f, cut2, c) if c else lpm(hv, f, cut2)
        row[name] = (round(r["estimate"], 4), [round(x, 3) for x in r["ci"]]); ADJ[h] = r if name == "all four" else ADJ.get(h, r)
    print(h, f, row)
print("\\nsame check on discovery (sector only):", {f: round(lpm_adjusted(dv, D["issuers"], f, cut2, ("sector",))["estimate"], 4) for f in LOCKED.values()})''')
md("# Ways I might be fooling myself\n\n1. **Search over 45 features** (discovery). Permutation test below.\n2. **Timing / leakage.** A real predictor should work only at the right horizon.\n3. **Near-perfect separation of f_14.**\n4. **Selection and survivorship.**")
code('''rng = np.random.default_rng(0)
X = dv[FEATURES].copy(); X = (X - X.mean()) / X.std(); yv = dv.deterioration_6m.values.astype(float)
def slopes(y):
    m = X.notna().values; Xc = np.where(m, X.values - np.nanmean(X.values, 0), 0)
    return (Xc * (y - y.mean())[:, None]).sum(0) / (Xc ** 2).sum(0) * 100
obs = slopes(yv); mx = []
for _ in range(200):
    yp = dv.assign(y=yv).groupby("month").y.transform(lambda s: rng.permutation(s.values)).values; mx.append(np.abs(slopes(yp)).max())
mx = np.array(mx)
print("null max |slope| 95th pct: %.2f pp (max %.2f)" % (np.percentile(mx, 95), mx.max()))
for f in ["f_14", "f_39"]:
    s = obs[FEATURES.index(f)]; print(f, "discovery slope %.2f pp -> family-wise permutation p = %.3f" % (s, (mx >= abs(s)).mean()))
oo = out.set_index(["issuer_id", "month"]).deterioration_6m
print("\\ntiming placebo (pp per SD at other outcome-month offsets):")
for k in (-6, -3, 0, 3, 6, 12):
    d = dv.copy(); m = pd.PeriodIndex(d.month, freq="M") + k
    d["deterioration_6m"] = oo.reindex(pd.MultiIndex.from_arrays([d.issuer_id, m.strftime("%Y-%m")])).values
    print("  offset %+d" % k, {f: round(lpm(d.dropna(subset=["deterioration_6m"]), f, cut)["estimate"] * 100, 1) for f in ("f_14", "f_39")})
q = pd.qcut(dv.f_14, 10, duplicates="drop"); print("\\nf_14 decile deterioration rates:", dv.groupby(q).deterioration_6m.mean().round(2).tolist())
print("rate with an event vs without (all issuer-months):", out.assign(has=out.set_index(["issuer_id", "month"]).index.isin(ev.set_index(["issuer_id", "month"]).index)).groupby("has").deterioration_6m.mean().round(3).to_dict())''')
md("Notes on the output above:\n\n- `f_14` survives the permutation test over all 45 features and predicts only the same-month label, but the top three deciles are 100% deteriorated. No real credit signal is that clean, so it may be a planted effect in synthetic data or a feature recorded alongside the label. I cannot tell which from the data. The holdout replicates it, which rules out noise but not leakage.\n- `f_39` was weak in discovery (family-wise permutation p about 0.27). It passes in the holdout, but with a lower confidence bound of about 0.04 and a late-period slope below 0.05 it is a marginal pass.\n- Events cluster where deterioration happens (rate about 44% versus 13% without an event), so conclusions apply to issuer-months with an event.\n- Defaulted issuers have no outcome rows. Counting them as deteriorated barely moves the estimates (see holdout robustness).\n- Mechanism labels were chosen after seeing the data.")
md("# Conclusions\n\n- **H1 (`f_14`, liquidity): supported.** Holdout slope of about +0.32 probability per SD, far above the 0.05 bar, stable across every robustness cut.\n- **H2 (`f_39`, operational): passes the locked test but is not supported once sector is controlled.** Holdout slope +0.067 (95% CI roughly 0.04 to 0.10) clears the 0.05 bar, but with sector fixed effects it falls to about +0.03 (not significant), and the same happens in discovery. `f_39` mostly proxies which sectors deteriorate more, so I record it as inconclusive (concentration).\n- Both results come from one pre-registered test each on a holdout with no overlap with discovery. The unexplained perfect separation of `f_14` is the main reason to treat the size of H1 with caution.")
code('''def gate(h):  # decision rule: the locked test must pass AND still pass (>= 0.05, one-sided p < 0.025) with all four fixed effects
    a = ADJ[h]; return ("keep", "") if (HOLDOUT[h]["decision"] == "keep" and a["estimate"] >= MIN_EFFECT and a["p_one_sided"] < ALPHA) else ("inconclusive", "concentration")
hyp = [dict(hypothesis_id=h, feature=LOCKED[h], estimate=round(r["estimate"], 6), ci=[round(c, 6) for c in r["ci"]],
            p_value=float(r["p_one_sided"]), n=r["n"], decision=gate(h)[0], kill_reason=gate(h)[1],
            locked_test_passed=r["decision"] == "keep", adjusted_estimate=round(ADJ[h]["estimate"], 6),
            adjusted_ci=[round(c, 6) for c in ADJ[h]["ci"]]) for h, r in HOLDOUT.items()]
checks = dict(dedup_rows_removed=int(D["dedup_removed"] + H["dedup_removed"]),
              sentinel_handling="-999 treated as missing (NaN) before winsorizing/standardizing; among duplicate event_ids the copy with fewest missing features is kept",
              timezone_normalized=True, exits_included=True,
              exits_note="primary tests use issuers with outcome rows only; robustness counts defaulted/distressed-exchange issuers without outcome rows as deteriorated",
              point_in_time="only events with published_at inside their event month are used; later-published revisions excluded",
              issuer_id_normalized="lowercased and stripped before all joins")
(ROOT / "submit" / "results.json").write_text(json.dumps(dict(hypotheses=hyp, data_checks=checks), indent=2))
print(json.dumps(dict(hypotheses=hyp, data_checks=checks), indent=2))''')
nb.cells=C; nb.metadata['kernelspec']=dict(name='python3',display_name='Python 3',language='python')
nbf.write(nb,'notebooks/analysis.ipynb')
