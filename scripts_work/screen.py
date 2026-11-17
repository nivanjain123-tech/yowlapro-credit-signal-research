import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, statsmodels.api as sm
from takehome.helpers import *
p=load_pack(); e=clean_events(p['events']); o=p['outcomes']; iss=p['issuers']
e=e[~e.issuer_id.str.contains('canary')].copy()
F=[c for c in e if c.startswith('f_')]
for c in F: e[c]=winsorize(e[c].astype(float))
def build(align):
    d=e.copy()
    d['dm']=d['month'] if align=='event' else d.published_at.dt.strftime('%Y-%m')
    d=d.merge(o,left_on=['issuer_id','dm'],right_on=['issuer_id','month'],how='inner',suffixes=('','_o')).merge(iss,on='issuer_id',how='left')
    return d
def screen(d,F):
    rows=[]
    for f in F:
        s=d[[f,'deterioration_6m','issuer_id']].dropna()
        z=(s[f]-s[f].mean())/s[f].std()
        r=sm.OLS(s.deterioration_6m,sm.add_constant(z)).fit(cov_type='cluster',cov_kwds={'groups':s.issuer_id})
        rows.append((f,len(s),r.params.iloc[1]*100,r.pvalues.iloc[1]))
    t=pd.DataFrame(rows,columns=['f','n','pp_per_sd','p']).set_index('f')
    t['q_bh']=sm.stats.multipletests(t.p,method='fdr_bh')[1]
    return t
if __name__=='__main__':
    for al in ('published','event'):
        d=build(al); t=screen(d,F); print(al,len(d),d.issuer_id.nunique(),d.deterioration_6m.mean().round(3))
        print(t.reindex(t.pp_per_sd.abs().sort_values(ascending=False).index).head(8).round(4))
