import numpy as np
X=np.memmap('1fme.f32',dtype=np.float32,mode='r').reshape(-1,504,3)
F=len(X)
def center(a): return a-a.mean(1,keepdims=True)
def kabsch_rmsd(P,Q):  # P,Q: (n,N,3) centered
    H=np.einsum('nia,nib->nab',P,Q)
    U,S,Vt=np.linalg.svd(H)
    d=np.sign(np.linalg.det(U@Vt))
    S[:,2]*=d
    e=(P**2).sum((1,2))+(Q**2).sum((1,2))-2*S.sum(1)
    return np.sqrt(np.maximum(e,0)/P.shape[1])
ref=center(X[0:1].astype(np.float64))
mean=np.zeros((F,3)); rg=np.zeros(F); r1=np.zeros(F)
ffr=np.full(F,np.nan); med=np.full(F,np.nan); mx=np.full(F,np.nan); com_step=np.full(F,np.nan)
C=2000; prev=None
for s in range(0,F,C):
    a=X[s:s+C].astype(np.float64)
    m=a.mean(1); mean[s:s+C]=m
    c=a-m[:,None,:]
    rg[s:s+C]=np.sqrt((c**2).sum(2).mean(1))
    r1[s:s+C]=kabsch_rmsd(c,np.broadcast_to(ref,c.shape))
    full=a if prev is None else np.concatenate([prev,a])
    off=s if prev is None else s-1
    d=np.linalg.norm(full[1:]-full[:-1],axis=2)
    idx=np.arange(off+1,off+len(full))
    med[idx]=np.median(d,1); mx[idx]=d.max(1)
    cf=full-full.mean(1,keepdims=True)
    ffr[idx]=kabsch_rmsd(cf[1:],cf[:-1])
    prev=a[-1:]
np.savez('1fme_signals.npz',mean=mean,rg=rg,r1=r1,ffr=ffr,med=med,mx=mx)
print("done")
