import sys; sys.path.insert(0,'scripts_work')
from screen import *
e['lag']=(e.published_at-e.event_time_utc).dt.days
e['pub_in_month']=e.published_at.dt.strftime('%Y-%m')==e.month
print(e.groupby('source_class').agg(n=('lag','size'),late=('pub_in_month',lambda s:(~s).mean()),medlag=('lag','median')).round(2))
print(e.lag.groupby(pd.cut(e.lag,[-1,0,3,10,30,60,90,130])).size().to_dict())
d=build('event'); d['pub_ok']=d.published_at.dt.strftime('%Y-%m')==d.month
print(d.groupby('pub_ok').deterioration_6m.agg(['mean','size']).round(3).to_dict())
v=d[d.pub_ok]; print('PIT-valid',len(v),v.issuer_id.nunique(),v.deterioration_6m.mean().round(3), 'dups per issuer-month',v.duplicated(['issuer_id','month']).sum())
t=screen(v,F); t['sig']=t.q_bh<.1
print(t.reindex(t.pp_per_sd.abs().sort_values(ascending=False).index).head(10).round(4))
# stability for top features
half=v.dm<'2020-09'
for f in t.reindex(t.pp_per_sd.abs().sort_values(ascending=False).index).head(8).index:
    a=screen(v[half],[f]).pp_per_sd.iloc[0]; b=screen(v[~half],[f]).pp_per_sd.iloc[0]
    sec={s:round(screen(g,[f]).pp_per_sd.iloc[0],1) for s,g in v.groupby('size_bucket')}
    print(f,round(a,1),round(b,1),sec)
v.to_pickle('/tmp/_v.pkl') if False else None
