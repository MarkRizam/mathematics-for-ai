"""Fixed synthetic protocols; run from repository root."""
import json
import platform
from pathlib import Path
import numpy as np
from .core import lu, qr, solve, gradient, optimize, Value

SEEDS = (11,22,33)


def run():
    rows=[]
    def add(exp,seed,config,metric,ours,reference):
        rows.append(dict(experiment=exp,seed=seed,configuration=config,metric=metric,ours=float(ours),reference=float(reference)))
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        a=rng.normal(size=(6,6))
        p,l,u=lu(a)
        add('EXP-001',seed,'pivoted LU','relative residual',np.linalg.norm(p@a-l@u)/np.linalg.norm(a),0)
        a=rng.normal(size=(10,4))
        q,r=qr(a); q0,r0=np.linalg.qr(a)
        add('EXP-002',seed,'QR reconstruction','relative residual',np.linalg.norm(q@r-a)/np.linalg.norm(a),np.linalg.norm(q0@r0-a)/np.linalg.norm(a))
        add('EXP-002',seed,'QR orthogonality','Frobenius error',np.linalg.norm(q.T@q-np.eye(4)),np.linalg.norm(q0.T@q0-np.eye(4)))
        q1,_=np.linalg.qr(rng.normal(size=(5,5))); q2,_=np.linalg.qr(rng.normal(size=(5,5)))
        true=rng.normal(size=5)
        for condition in (1,1e4,1e8):
            a=q1@np.diag(np.geomspace(1,1/condition,5))@q2.T; b=a@true
            add('EXP-003',seed,f'condition={condition:g}','relative solution error',np.linalg.norm(solve(a,b)-true)/np.linalg.norm(true),np.linalg.norm(np.linalg.solve(a,b)-true)/np.linalg.norm(true))
        x=rng.uniform(-1,1,size=3)
        for h in (1e-1,1e-3,1e-5,1e-7,1e-9):
            add('EXP-004',seed,f'h={h:g}','gradient L2 error',np.linalg.norm(gradient(lambda t:np.sin(t).sum(),x,h)-np.cos(x)),0)
        x=rng.normal(size=2)
        f=lambda t: .5*(t[0]**2+10*t[1]**2)
        grad=lambda t: t*np.array([1,10])
        for method in ('gd','momentum','adam'):
            for rate in (.01,.05):
                _,history=optimize(f,grad,x,method,rate,200)
                add('EXP-005',seed,f'{method}, rate={rate}','final objective gap',history[-1],0)
        x,y=rng.uniform(-1,1,size=2); a,b=Value(x),Value(y); z=(a*b+a).tanh()+a*a; z.backward()
        t=1-np.tanh(x*y+x)**2; truth=np.array([t*(y+1)+2*x,t*x]); fd=gradient(lambda v:np.tanh(v[0]*v[1]+v[0])+v[0]**2,[x,y])
        add('EXP-006',seed,'reverse mode vs central difference','gradient L2 error',np.linalg.norm([a.grad-truth[0],b.grad-truth[1]]),np.linalg.norm(fd-truth))
        a=rng.normal(size=(6,4)); u,s,vt=np.linalg.svd(a,full_matrices=False)
        for rank in (1,2,3,4):
            add('EXP-007',seed,f'SVD rank={rank}','relative reconstruction error',np.linalg.norm(a-(u[:,:rank]*s[:rank])@vt[:rank])/np.linalg.norm(a),np.linalg.norm(s[rank:])/np.linalg.norm(s))
        a=rng.normal(size=(4,4)); a=a+a.T; vals,vec=np.linalg.eigh(a)
        add('EXP-008',seed,'symmetric eigenpairs','relative eigenpair residual',np.linalg.norm(a@vec-vec*vals)/np.linalg.norm(a),0)
    return rows


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path.cwd(); (root/'results').mkdir(exist_ok=True); (root/'figures').mkdir(exist_ok=True)
    a,b=run(),run()
    assert len(a)==len(b)==69
    for left,right in zip(a,b):
        for key,val in left.items():
            if isinstance(val,float): np.testing.assert_allclose(val,right[key],rtol=0,atol=1e-10)
            else: assert val==right[key]
    summary=[]
    for exp in sorted({r['experiment'] for r in a}):
        rows=[r for r in a if r['experiment']==exp]; configs=list(dict.fromkeys(r['configuration'] for r in rows))
        fig,ax=plt.subplots(figsize=(9,4.5)); xx=np.arange(len(configs))
        for config in configs:
            group=[r for r in rows if r['configuration']==config]
            summary.append(dict(experiment=exp,configuration=config,metric=group[0]['metric'],n=len(group),**{f'{key}_{stat}':float(fn([r[key] for r in group])) for key in ('ours','reference') for stat,fn in [('mean',np.mean),('sd',lambda v:np.std(v,ddof=1))]}))
        for i,key in enumerate(('ours','reference')):
            group=[s for s in summary if s['experiment']==exp]
            ax.bar(xx+(i-.5)*.35,[s[key+'_mean'] for s in group],.35,yerr=[s[key+'_sd'] for s in group],capsize=3,label=key)
        ax.set_xticks(xx,configs,rotation=25,ha='right'); ax.set_ylabel(rows[0]['metric'] if exp!='EXP-002' else 'Error (see configuration)'); ax.set_title(exp+' — mean and sample SD, 3 seeds'); ax.legend(); fig.tight_layout()
        fig.savefig(root/'figures'/f'{exp}.svg'); plt.close(fig)
    payload=dict(project='AIR-002',seeds=SEEDS,environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.system()),reproducibility=dict(runs=2,atol=1e-10,rtol=0,status='pass'),records=a,summary=summary)
    (root/'results'/'benchmarks.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(f'{len(a)} records, {len(summary)} configurations, 8 experiments; two runs matched')


if __name__=='__main__': main()
