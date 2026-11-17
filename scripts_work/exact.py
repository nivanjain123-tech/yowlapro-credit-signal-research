import warnings; warnings.filterwarnings('ignore'); import sys, json; sys.path.insert(0,'src')
from takehome.analysis import *
D=prepare('data'); H=prepare('/Users/nivanvishaljain/Desktop/YowlaPro/holdout')
cut=clip_points(D['events_clean'],['f_14','f_39']); dv=pit_sample(D['events_clean'],D['outcomes']); hv=pit_sample(H['events_clean'],H['outcomes'])
hx=with_exits_as_deteriorated(H['events_clean'],H['outcomes'],H['exits'])
print('dedup removed disc/hold',D['dedup_removed'],H['dedup_removed'], '| rows added by exit sensitivity',len(hx)-len(hv))
for h,f in LOCKED.items():
    r=lpm(hv,f,cut); print(h,f,{k:(float(v) if not isinstance(v,list) else [round(x,4) for x in v]) for k,v in r.items()},verdict(r))
    print('  disc',{k:(round(v,5) if not isinstance(v,list) else v) for k,v in lpm(dv,f,cut).items() if k!='ci'})
    print('  exits-as-bad sens',{k:(round(v,4) if not isinstance(v,list) else [round(x,3) for x in v]) for k,v in lpm(hx,f,cut).items()})
# overlap of holdout and discovery issuer-months
ov=set(zip(dv.issuer_id,dv.month))&set(zip(hv.issuer_id,hv.month)); print('overlapping issuer-month rows',len(ov),'of holdout',len(hv))
