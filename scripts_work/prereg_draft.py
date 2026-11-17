PLAN = ("Holdout, events published within their event month, joined to deterioration_6m on issuer-month. Linear probability model of "
        "deterioration_6m on standardized {f} (clipped at discovery 1st/99th pct), SEs clustered by issuer; slope = effect per 1 SD in probability. "
        "Pass if slope >= 0.05 and one-sided p < 0.025 (Bonferroni, 2 tests).")
FALS = ("On the holdout, the {f} slope per 1 SD is below 0.05, or is not positive, or the one-sided issuer-clustered p-value is above 0.025.")
H = {
"H1": dict(pred="f_14", mech="liquidity",
 obs="Discovery, events published within their event month (n=696, 338 issuers): deterioration is ~10-20% for f_14 below ~0.25, ~67% for 0.25-0.5 and ~100% above ~0.5 (+33pp per SD). Same sign in both halves, all sectors and sizes. Near-separation; absent at other outcome months.",
 expl="A higher f_14 marks tighter funding or cash strain, which raises the chance of a downgrade or missed payment within six months.",
 plan=PLAN.format(f="f_14"), fals=FALS.format(f="f_14")),
"H2": dict(pred="f_39", mech="operational",
 obs="Discovery, same sample (n=696): higher f_39 goes with more 6m deterioration, +5.2pp per SD (cluster p=0.004). Weak: family-wise permutation p=0.27 over 45 features; strongest in news_wire. Positive in both halves and all size buckets. Treated as a modest, exploratory-strength signal.",
 expl="A higher f_39 marks weaker operating performance (margins or revenue), which erodes credit quality over the following six months.",
 plan=PLAN.format(f="f_39"), fals=FALS.format(f="f_39")),
}
if __name__=="__main__":
    lim=dict(obs=400,plan=400,fals=300)
    for k,h in H.items():
        print(k,h['pred'],h['mech'],{f:f"{len(h[f])}/{l}" for f,l in lim.items()})
