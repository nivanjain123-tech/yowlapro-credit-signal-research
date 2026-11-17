import warnings; warnings.filterwarnings('ignore')
import duckdb, numpy as np, pandas as pd, statsmodels.api as sm
H='/Users/nivanvishaljain/Desktop/YowlaPro/holdout/'; D='data/'
# fully independent path: DuckDB SQL on the parquet files, no takehome code
q=lambda p,s:duckdb.sql(s.replace('{p}',p)).df()
def load(p):
    e=q(p,"select * from read_parquet('{p}events.parquet') where lower(issuer_id) not like '%canary%'")
    F=[f'f_{i:02d}' for i in range(1,46)]
    e['nm']=(e[F]==-999).sum(1)+e[F].isna().sum(1); e=e.sort_values('nm',kind='stable').drop_duplicates('event_id')
    e[F]=e[F].replace(-999,np.nan); e['issuer_id']=e.issuer_id.str.lower()
    e['pub_month']=pd.to_datetime(e.published_at,utc=True).dt.strftime('%Y-%m')
    o=q(p,"select lower(issuer_id) issuer_id, month, deterioration_6m y from read_parquet('{p}outcomes.parquet')")
    return e[e.pub_month==e.month].merge(o,on=['issuer_id','month']),F
d,_=load(D); h,F=load(H)
print('holdout n',len(h),'issuers',h.issuer_id.nunique(),'| overlap issuer-months with discovery',len(set(zip(d.issuer_id,d.month))&set(zip(h.issuer_id,h.month))))
cut={f:d[f].quantile([.01,.99]).values for f in ('f_14','f_39')}
def fit(df,f,clip=True):
    s=df[[f,'y','issuer_id']].dropna(); x=s[f].clip(*cut[f]) if clip else s[f]; x=(x-x.mean())/x.std()
    r=sm.OLS(s.y.astype(float),sm.add_constant(x)).fit(cov_type='cluster',cov_kwds={'groups':s.issuer_id}); return s,x,r
rng=np.random.default_rng(7)
for f in ('f_14','f_39'):
    s,x,r=fit(h,f); b=r.params.iloc[1]
    # wild-free issuer-cluster bootstrap and permutation of y within month (holdout, single pre-registered feature)
    ids=s.issuer_id.unique(); grp={i:g.index.values for i,g in s.groupby('issuer_id')}; xs=pd.Series(x.values,index=s.index); ys=s.y.astype(float)
    bs=[]
    for _ in range(1000):
        idx=np.concatenate([grp[i] for i in rng.choice(ids,len(ids))]); xx=xs.loc[idx]; yy=ys.loc[idx]; bs.append(np.cov(xx,yy)[0,1]/xx.var())
    ph=[]
    hm=h.loc[s.index,'month'].values
    for _ in range(1000):
        yp=ys.groupby(hm).transform(lambda v: rng.permutation(v.values)); ph.append(np.cov(xs,yp)[0,1]/xs.var())
    print(f,'slope',round(b,4),'cluster CI',np.round(r.conf_int().iloc[1].values,4),'| bootstrap 95% CI',np.round(np.percentile(bs,[2.5,97.5]),4),'| permutation p (one-sided)',(np.sum(np.array(ph)>=b)+1)/1001,'| P(boot slope<0.05)',round(np.mean(np.array(bs)<0.05),3))
    # extra-precise alternative: standardize with discovery mean/SD instead of holdout SD
    xd=(s[f].clip(*cut[f])-d[f].clip(*cut[f]).mean())/d[f].clip(*cut[f]).std(); r2=sm.OLS(s.y.astype(float),sm.add_constant(xd)).fit(cov_type='cluster',cov_kwds={'groups':s.issuer_id}); print('   slope per DISCOVERY SD',round(r2.params.iloc[1],4))
    # month fixed effects and sector/size controls (precision)
    iss=q(H,"select lower(issuer_id) issuer_id,sector,size_bucket from read_parquet('{p}issuers.parquet')"); t=s.join(h[['month']]).merge(iss,on='issuer_id',how='left')
    X=pd.get_dummies(t[['month','sector','size_bucket']],drop_first=True).astype(float); X.insert(0,'x',x.values); X=sm.add_constant(X)
    r3=sm.OLS(t.y.astype(float).values,X.values).fit(cov_type='cluster',cov_kwds={'groups':t.issuer_id.values}); print('   with month+sector+size fixed effects: slope',round(r3.params[1],4),'CI',np.round(r3.conf_int()[1],4))
