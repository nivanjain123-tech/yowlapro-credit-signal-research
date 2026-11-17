P = ("On the holdout, I'll keep only events published in the same month they happened, and match them to that month's deterioration_6m for the same issuer. "
     "I'll fit a simple linear probability model of deterioration_6m on standardized {f}, with errors clustered by issuer. I'll call it a pass if the effect per 1 SD is at least 0.05 and the one-sided p is below 0.025 (two tests).")
Fz = "It fails if, on the holdout, the effect of {f} per 1 SD is under 0.05, is not positive, or the one-sided issuer-clustered p-value is above 0.025."
H = {
"H1": dict(obs="In the discovery data (696 events published in their own month, 338 issuers), f_14 is strongly tied to deterioration. Below about 0.25 only 10-20% deteriorate, from 0.25 to 0.5 about two thirds do, and above 0.5 essentially all do (+33pp per SD). It held in both halves, all sectors and sizes. It's suspiciously clean, so I'm cautious.",
 expl="A higher f_14 looks like tighter funding or cash strain, which makes a downgrade or missed payment within six months more likely.",
 plan=P.format(f="f_14"), fals=Fz.format(f="f_14")),
"H2": dict(obs="In the same discovery sample, a higher f_39 goes with more deterioration, about +5.2pp per SD (cluster p=0.004). I don't trust it much: after allowing for testing 45 features the permutation p is 0.27, and it is strongest in news_wire events. Still, it is positive in both halves and all size buckets, so I'm testing it.",
 expl="A higher f_39 looks like weaker operating performance (margins or revenue), which gradually erodes credit quality over the next six months.",
 plan=P.format(f="f_39"), fals=Fz.format(f="f_39")),
}
for k,h in H.items():
    print(k,{f:f"{len(h[f])}/{l}" for f,l in dict(obs=400,expl=0,plan=400,fals=300).items() if l}, len(h['expl']))
    for f in ('obs','expl','plan','fals'): print(f"[{f}] {h[f]}")
