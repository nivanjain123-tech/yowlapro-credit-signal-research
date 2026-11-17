import warnings; warnings.filterwarnings('ignore'); import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd, statsmodels.api as sm
from takehome.analysis import *
D=prepare('data'); H=prepare('/Users/nivanvishaljain/Desktop/YowlaPro/holdout')
cut=clip_points(D['events_clean'],['f_14','f_39']); dv=pit_sample(D['events_clean'],D['outcomes'])
def run(v,iss,f,ctrl):
    t=v.merge(iss[['issuer_id','sector','size_bucket']],on='issuer_id',how='left')[[f,'deterioration_6m','issuer_id','month','sector','size_bucket','source_class']].dropna(subset=[f,'deterioration_6m'])
    x=t[f].clip(*cut[f]); x=(x-x.mean())/x.std()
    X=pd.DataFrame({'x':x.values})
    if ctrl: X=pd.concat([X,pd.get_dummies(t[ctrl],drop_first=True).astype(float).reset_index(drop=True)],axis=1)
    r=sm.OLS(t.deterioration_6m.astype(float).values,sm.add_constant(X).values).fit(cov_type='cluster',cov_kwds={'groups':t.issuer_id.values})
    return round(r.params[1],4),np.round(r.conf_int()[1],3).tolist(),round(r.pvalues[1],5)
for name,v,iss in (('HOLDOUT',pit_sample(H['events_clean'],H['outcomes']),H['issuers']),('DISCOVERY',dv,D['issuers'])):
    print(name)
    for f in ('f_14','f_39'):
        print(' ',f,{c:run(v,iss,f,cc) for c,cc in [('none',[]),('month',['month']),('sector',['sector']),('size',['size_bucket']),('source',['source_class']),('all',['month','sector','size_bucket','source_class'])]})
