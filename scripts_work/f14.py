import sys; sys.path.insert(0,'scripts_work')
from screen import *
d=build('published'); d['lag']=(d.published_at-d.event_time_utc).dt.days
print(pd.qcut(d.f_14,10,duplicates='drop').pipe(lambda q:d.groupby(q).deterioration_6m.agg(['mean','size'])).round(3))
print(d[['f_14','f_08','lag','deterioration_6m','spread_change_6m']].corr().round(2))
def eff(g,f='f_14'):
    s=g[[f,'deterioration_6m','issuer_id']].dropna()
    if len(s)<40 or s.deterioration_6m.nunique()<2: return (len(s),np.nan)
    z=(s[f]-s[f].mean())/s[f].std(); r=sm.OLS(s.deterioration_6m,sm.add_constant(z)).fit(); return (len(s),round(r.params.iloc[1]*100,1))
for k in ['sector','size_bucket','source_class']: print(k,{a:eff(g) for a,g in d.groupby(k)})
d['half']=np.where(d.dm<'2020-10','early','late'); print({a:eff(g) for a,g in d.groupby('half')})
d['late_pub']=d.lag>30; print({a:eff(g) for a,g in d.groupby('late_pub')})
print(d.groupby('late_pub').deterioration_6m.mean().to_dict(), d.groupby('source_class').deterioration_6m.mean().round(3).to_dict())
print(d.groupby('dm').deterioration_6m.agg(['mean','size']).round(2).T.to_string())
print(d.f_14.describe().round(2).to_dict()); print(d.groupby(d.f_14>d.f_14.quantile(.9)).deterioration_6m.mean().round(3).to_dict())
