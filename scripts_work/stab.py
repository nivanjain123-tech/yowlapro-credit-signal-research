import sys; sys.path.insert(0,'scripts_work')
from screen import *
d=build('event'); v=d[d.published_at.dt.strftime('%Y-%m')==d.month].copy().reset_index(drop=True)
X=v[F].copy(); X=(X-X.mean())/X.std(); y=v.deterioration_6m.values.astype(float)
g=v.issuer_id.values; ids=np.unique(g); idx={i:np.where(g==i)[0] for i in ids}
rng=np.random.default_rng(0); B=300
def slopes(rows):
    Xs=X.values[rows]; ys=y[rows]; m=~np.isnan(Xs)
    Xc=np.where(m,Xs,0); n=m.sum(0); mx=Xc.sum(0)/n; yc=ys[:,None]-ys.mean()
    xc=np.where(m,Xs-mx,0); return (xc*yc).sum(0)/(xc**2).sum(0)*100
full=slopes(np.arange(len(v)))
S=[]; 
for b in range(B):
    s=rng.choice(ids,len(ids)); rows=np.concatenate([idx[i] for i in s]); S.append(slopes(rows))
S=np.array(S); res=pd.DataFrame({'full':full,'boot_mean':S.mean(0),'lo5':np.percentile(S,5,0),'p_pos':(S>0).mean(0),
 'p_ge4':(S>=4).mean(0)},index=F)
# selection frequency: how often is feature among top-3 positive slopes in a bootstrap, excluding f_14
top3=pd.Series(0,index=F)
for s in S: top3[np.array(F)[np.argsort(-s)[:4]]]+=1
res['top4_freq']=top3/B
# split-half out-of-sample: select on half the issuers, evaluate slope on other half
oos=pd.DataFrame(index=F,data={'sel':0,'oos_sum':0.0}); 
for r in range(200):
    h=rng.permutation(ids); a=np.concatenate([idx[i] for i in h[:len(h)//2]]); b=np.concatenate([idx[i] for i in h[len(h)//2:]])
    sa,sb=slopes(a),slopes(b); top=np.argsort(-sa)[:4]
    for t in top: oos.iloc[t,0]+=1; oos.iloc[t,1]+=sb[t]
res['sel_freq']=oos.sel/200; res['oos_slope_when_sel']=oos.oos_sum/oos.sel.replace(0,np.nan)
print(res.sort_values('full',ascending=False).head(10).round(2))
