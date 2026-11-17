import warnings; warnings.filterwarnings('ignore')
import duckdb, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
c=duckdb.connect('data/research.duckdb',read_only=True)   # independent path: raw SQL, no helpers.py
ev=c.execute("""
 SELECT * EXCLUDE rn FROM (SELECT *, row_number() OVER (PARTITION BY event_id ORDER BY
   (SELECT 0)) rn FROM events WHERE lower(issuer_id) NOT LIKE '%canary%') WHERE rn=1""").df()
# duplicates: prefer copy with fewest -999 (do in pandas, independent of helpers)
raw=c.execute("select * from events where lower(issuer_id) not like '%canary%'").df()
F=[f'f_{i:02d}' for i in range(1,46)]
raw['nm']=(raw[F]==-999).sum(1)+raw[F].isna().sum(1)
raw=raw.sort_values('nm',kind='stable').drop_duplicates('event_id'); raw['issuer_id']=raw.issuer_id.str.lower()
raw[F]=raw[F].replace(-999,np.nan)
loc=pd.to_datetime(raw.event_time)
raw['utc']=[t.tz_localize(z).tz_convert('UTC') for t,z in zip(loc,raw.event_tz)]
raw['pub']=pd.to_datetime(raw.published_at,utc=True)
print('month label == local event month:',(loc.dt.strftime('%Y-%m')==raw.month).mean().round(4),'| == utc event month:',(raw.utc.dt.strftime('%Y-%m')==raw.month).mean().round(4))
ok=raw[raw.pub.dt.strftime('%Y-%m')==raw.month.values].copy()
o=c.execute("select lower(issuer_id) issuer_id, month, deterioration_6m y from outcomes").df()
ok=ok.merge(o,on=['issuer_id','month'])
print('n',len(ok),'issuers',ok.issuer_id.nunique(),'events/ issuer-month dup rows',ok.duplicated(['issuer_id','month']).sum(),'| deterioration rate',ok.y.mean().round(3))
# selection check: is the event-row base rate unusual vs all outcomes in same months?
mm=o[o.month.isin(ok.month.unique())]; print('base rate all issuer-months same months',mm.y.mean().round(3),'| issuers with an event that month vs without:',
  o.assign(has=o.set_index(['issuer_id','month']).index.isin(raw.set_index(['issuer_id','month']).index)).groupby('has').y.mean().round(3).to_dict())
def z(s,w=True):
    if w: s=s.clip(*s.quantile([.01,.99]))
    return (s-s.mean())/s.std()
def lpm(df,f,w=True):
    s=df[[f,'y','issuer_id']].dropna(); x=z(s[f],w)
    r=sm.OLS(s.y,sm.add_constant(x)).fit(cov_type='cluster',cov_kwds={'groups':s.issuer_id}); return r.params.iloc[1]*100,r.pvalues.iloc[1]
def logit_ame(df,f):
    s=df[[f,'y','issuer_id']].dropna(); s=s.assign(x=z(s[f]))
    r=smf.logit('y~x',s).fit(disp=0); p=r.predict(s.assign(x=s.x+0.5))-r.predict(s.assign(x=s.x-0.5)); return p.mean()
print('\n--- independent reproduction (LPM %pt/SD, p | logit AME/SD)')
for f in ['f_14','f_39','f_05','f_06']: print(f,[round(v,4) for v in lpm(ok,f)],round(logit_ame(ok,f),3))
# issuer-bootstrap CI for logit AME
rng=np.random.default_rng(1); ids=ok.issuer_id.unique(); grp={i:g for i,g in ok.groupby('issuer_id')}
for f in ['f_14','f_39','f_05']:
    bs=[]
    for _ in range(300):
        d=pd.concat([grp[i] for i in rng.choice(ids,len(ids))]); bs.append(logit_ame(d,f))
    print(f,'bootstrap AME 95% CI',np.percentile(bs,[2.5,97.5]).round(3))
print('\n--- permutation (shuffle y within month): family-wise over 45 features')
obs=np.array([abs(lpm(ok,f)[0]) for f in F]); mx=[]
for _ in range(200):
    d=ok.copy(); d['y']=d.groupby('month').y.transform(lambda s: rng.permutation(s.values)); mx.append(max(abs(lpm(d,f)[0]) for f in F))
mx=np.array(mx); print('max |slope| under null: 95th pct',np.percentile(mx,95).round(2),'max',mx.max().round(2))
for f,v in zip(F,obs):
    if f in('f_14','f_39','f_05','f_06'): print(f,'obs |slope|',v.round(2),'FWER p',((mx>=v).mean()).round(3))
print('\n--- timing placebo: does feature predict outcome at other month offsets? (LPM %pt/SD)')
oo=o.set_index(['issuer_id','month']).y
def shifted(k):
    d=ok.copy(); m=pd.PeriodIndex(d.month,freq='M')+k; d['y']=oo.reindex(pd.MultiIndex.from_arrays([d.issuer_id,m.strftime('%Y-%m')])).values; return d.dropna(subset=['y'])
for k in (-12,-6,-3,0,3,6,12):
    d=shifted(k); print('offset',k,'n',len(d),{f:round(lpm(d,f)[0],1) for f in ['f_14','f_39','f_05']})
print('\n--- sensitivity (f_14 / f_39 / f_05 slopes)')
def row(name,d,w=True): print(name,len(d),[round(lpm(d,f,w)[0],1) for f in ['f_14','f_39','f_05']])
row('baseline',ok); row('no winsor',ok,False)
row('one row/issuer-month (mean)',ok.groupby(['issuer_id','month'],as_index=False)[F+['y']].mean())
row('drop exited issuers',ok[~ok.issuer_id.isin(c.execute("select lower(issuer_id) from exits").df().iloc[:,0])])
for sc,g in ok.groupby('source_class'): row('  '+sc,g)
for h,g in ok.groupby(ok.month<'2020-09'): row('  early' if h else '  late',g)
iss=c.execute("select lower(issuer_id) issuer_id,sector,size_bucket from issuers").df(); okk=ok.merge(iss,on='issuer_id')
print('sectors f_14',{s:round(lpm(g,'f_14')[0],0) for s,g in okk.groupby('sector')},'n>=?',okk.groupby('sector').size().min())
print('f_39 sizes',{s:round(lpm(g,'f_39')[0],1) for s,g in okk.groupby('size_bucket')},'f_05 sizes',{s:round(lpm(g,'f_05')[0],1) for s,g in okk.groupby('size_bucket')})
q=pd.qcut(ok.f_14,10,duplicates='drop'); print('f_14 decile rates',ok.groupby(q).y.mean().round(2).tolist())
print('f_14 low-third vs high-third rate', ok.y[ok.f_14<ok.f_14.quantile(.33)].mean().round(2), ok.y[ok.f_14>ok.f_14.quantile(.67)].mean().round(2))
print('corr f_14 with other top features',ok[['f_14','f_39','f_05','f_06']].corr().round(2).iloc[0].to_dict())
