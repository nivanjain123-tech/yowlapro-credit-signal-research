"""Pre-registered confirmatory tests (H1=f_14, H2=f_39). Run once, plan fixed before holdout unlock."""
import warnings, json, sys; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from datetime import datetime, timezone
from scipy import stats
sys.path.insert(0,'src')
from takehome.helpers import load_pack, clean_events
HOLD='/Users/nivanvishaljain/Desktop/YowlaPro/holdout'
disc=load_pack('data'); hold=load_pack(HOLD)
de=clean_events(disc['events']); de=de[~de.issuer_id.str.contains('canary')]
he=clean_events(hold['events']); he=he[~he.issuer_id.str.contains('canary')]
cut={f:de[f].quantile([.01,.99]).values for f in ['f_14','f_39']}   # discovery clip points
v=he[he.published_at.dt.strftime('%Y-%m')==he.month].merge(hold['outcomes'],on=['issuer_id','month'],how='inner')
exited=set(hold['exits'].issuer_id)
print('holdout events',len(he),'| published in event month',int((he.published_at.dt.strftime('%Y-%m')==he.month).sum()),'| matched to outcomes',len(v),'| issuers',v.issuer_id.nunique(),'| deterioration rate',round(v.deterioration_6m.mean(),3))
def zf(s,f,clip=True,sd='holdout'):
    s=s.clip(*cut[f]) if clip else s; return (s-s.mean())/s.std()
def lpm(df,f,clip=True):
    s=df[[f,'deterioration_6m','issuer_id']].dropna(); x=zf(s[f],f,clip)
    r=sm.OLS(s.deterioration_6m.astype(float),sm.add_constant(x)).fit(cov_type='cluster',cov_kwds={'groups':s.issuer_id})
    b=r.params.iloc[1]; se=r.bse.iloc[1]; p2=r.pvalues.iloc[1]; p1=p2/2 if b>0 else 1-p2/2
    return dict(n=len(s),est=b,lo=b-1.96*se,hi=b+1.96*se,p_one=p1,p_two=p2)
def logit_ame(df,f):
    s=df[[f,'deterioration_6m']].dropna().assign(x=lambda d:zf(d[f],f)); s['y']=s.deterioration_6m
    try:
        r=smf.logit('y~x',s).fit(disp=0); return float((r.predict(s.assign(x=s.x+.5))-r.predict(s.assign(x=s.x-.5))).mean())
    except Exception as ex: return None
out={}
for h,f in [('H1','f_14'),('H2','f_39')]:
    r=lpm(v,f); ok=(r['est']>=0.05) and (r['p_one']<0.025); r['verdict']='PASS' if ok else 'FAIL'; out[h]=dict(feature=f,primary=r)
    rob={'logit_ame':logit_ame(v,f),'no_clip':lpm(v,f,False),'one_row_per_issuer_month':lpm(v.groupby(['issuer_id','month'],as_index=False)[[f,'deterioration_6m']].mean(),f),
         'drop_exited':lpm(v[~v.issuer_id.isin(exited)],f)}
    out[h]['robust']=rob
    out[h]['halves']={k:lpm(g,f) for k,g in v.groupby(v.month<'2022-07')}
    out[h]['by_size']={k:lpm(g.merge(hold['issuers'][['issuer_id','size_bucket']],on='issuer_id'),f) for k,g in v.merge(hold['issuers'][['issuer_id','size_bucket']],on='issuer_id').groupby('size_bucket')}
    out[h]['by_source']={k:lpm(g,f) for k,g in v.groupby('source_class')}
def fmt(r): return {k:(round(x,4) if isinstance(x,float) else x) for k,x in r.items()} if isinstance(r,dict) else r
for h,d in out.items():
    print('\n=====',h,d['feature'],'PRIMARY',fmt(d['primary']))
    for k,r in d['robust'].items(): print('  robust',k,fmt(r) if isinstance(r,dict) else round(r,3) if r is not None else None)
    print('  halves',{('early' if k else 'late'):(round(r['est'],3),r['n']) for k,r in d['halves'].items()})
    print('  size',{k:(round(r['est'],3),r['n']) for k,r in d['by_size'].items()}); print('  source',{k:(round(r['est'],3),r['n']) for k,r in d['by_source'].items()})
json.dump(out,open('scripts_work/holdout_results.json','w'),default=float,indent=1)
print('\nrun at (UTC):',datetime.now(timezone.utc).isoformat())
