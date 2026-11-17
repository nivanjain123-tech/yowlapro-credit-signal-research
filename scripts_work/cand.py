import sys; sys.path.insert(0,'scripts_work')
from screen import *
d=build('event'); v=d[d.published_at.dt.strftime('%Y-%m')==d.month].copy()
x=p['exits']; allv=e[e.published_at.dt.strftime('%Y-%m')==e.month]
ex=allv[allv.issuer_id.isin(x.issuer_id)]
print('events from exited issuers',len(ex),ex.issuer_id.nunique(),'of',allv.issuer_id.nunique(),'| missing from outcomes:',len(set(allv.issuer_id)-set(o.issuer_id)))
print('f_14 mean exited vs rest', ex.f_14.mean().round(2), allv[~allv.issuer_id.isin(x.issuer_id)].f_14.mean().round(2))
for f in ['f_14','f_39','f_06','f_05']:
    s=v[[f,'deterioration_6m','issuer_id']].dropna(); z=(s[f]-s[f].mean())/s[f].std()
    r=sm.Logit(s.deterioration_6m,sm.add_constant(z)).fit(disp=0,cov_type='cluster',cov_kwds={'groups':s.issuer_id})
    m=r.get_margeff(at='overall').summary_frame().iloc[0]
    print(f,len(s),'AME/SD',round(m['dy/dx'],3),'CI',round(m['Conf. Int. Low'],3),round(m['Cont. Int. Hi.'],3),'p',round(m['Pr(>|z|)'],4))
