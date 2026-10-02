import sys, time, pickle; sys.path.insert(0,'.'); from run_1fme_full import *
d=np.load('1fme_signals.npz'); F=len(d['rg']); frames=np.arange(1,F+1)
sig={'Mean Z':(d['mean'][:,2],frames),'RMSD from frame 1':(d['r1'],frames),
     'Frame-to-frame RMSD':(d['ffr'][1:],frames[1:]),'Radius of gyration':(d['rg'],frames)}
which=sys.argv[1]; res={}
for name,(s,fr) in sig.items():
    t=time.time()
    if which=='kcusum':
        cal=(fr>=CAL_FIRST)&(fr<=CAL_LAST); use=fr>=CAL_FIRST
        res[name]=run_kcusum(s,fr,cal,use=use,label=name)
        if name=='Mean Z':
            res['Mean Z all']=run_kcusum(s,fr,np.ones(len(s),bool),label='Mean Z, ref all frames (from frame 1)')
    elif which=='bocpd':
        b=run_bocpd(s,fr); res[name]=b; print(name,'first',b['first'],'after cal',b['first_after_cal'],'n flagged',len(b['flagged']),flush=True)
    elif which=='kliep':
        k=run_kliep(s,fr); res[name]=k; print(name,'first',k['first'],'thr %.3f'%k['thr'],'frac after %.3f'%k['frac_after'],flush=True)
    print('  %.0fs'%(time.time()-t),flush=True)
pickle.dump(res,open(f'res_{which}.pkl','wb'))
