"""
Verification of the designed operating point (Theorem 1, conditions D1-D6) and
generator of Supplement Fig. S3 (design_verification.png).

Verifies, on exact diagonalizations of the designed Gamma_g:
  (i)   D3 equalization:  |Re lambda_k - lstar|/lstar  ~ 1e-9
  (ii)  node placement error epsbar ~ 1/Delta_0  (three decades)
  (iii) extrapolation tail obeys  sum_{m>M} ||c_m||_1  <=  4*M*rbar^M + 8*epsbar
  (iv)  kernel matching exact (extended precision), realized tail <= bound

Instrumentation notes (bugs fixed relative to the first draft of this script):
  - node error uses nearest-ideal per-node matching, NOT sorted-angle lists
    (sorted comparison breaks under a global rotation);
  - all Vandermonde solves in mpmath (50 dps): double precision dies by M=14,
    the mathematics does not.
"""
import numpy as np, mpmath as mp
mp.mp.dps=50
GAMMA=0.1; PHI=np.pi/3

def design(M, Delta0_override=None, beta=1e-5, gamma=GAMMA, gg_frac=0.5, tau=10.0):
    """Designed device: D1 weak dressing, D2 angle lock, D3 equalized LO, D4 generic tau."""
    K=M; gamma_g=gg_frac*gamma/(K-1); lstar=(gamma+gamma_g)/(2*K)
    T=np.log(2*M)/lstar                                   # D5: rbar = 1/(2M)
    if Delta0_override is None:                           # D1 sized against 2nd-order Im shift
        Delta0=max((gamma+gamma_g)**2*(2*np.log(K)+2)*T/(K**2*beta),4*(gamma+gamma_g))
    else: Delta0=Delta0_override
    n=max(1,int(round((Delta0*T*M/(2*np.pi)-1)/M)))       # D2 snap: Delta0*T=2pi(nM+1)/M
    Delta0=2*np.pi*(n*M+1)/(M*T)
    k=np.arange(K); delta=Delta0*k
    alpha=np.sin((np.pi-PHI)/2+(delta+5.0)*tau/2)
    while np.min(np.abs(alpha))<1e-2:                     # D4: step off emitter nodes
        tau+=0.137; alpha=np.sin((np.pi-PHI)/2+(delta+5.0)*tau/2)
    alpha=alpha/np.linalg.norm(alpha)
    v=np.sqrt((2*lstar-gamma_g*alpha**2)/gamma)           # D3 (xi_0=0 here)
    G=1j*np.diag(delta)+(gamma/2)*np.outer(v,v)+(gamma_g/2)*np.outer(alpha,alpha)
    return dict(M=M,T=T,Delta0=Delta0,lstar=lstar,G=G)

def analyze(d,periods=3,seed=1):
    M=d["M"]; lam=np.linalg.eigvals(d["G"]); lam=lam[np.argsort(lam.imag)]
    x=np.exp(-lam*d["T"]); rbar=np.exp(-d["lstar"]*d["T"])
    re_spread=np.abs(lam.real-d["lstar"]).max()/d["lstar"]
    ideal=rbar*np.exp(-2j*np.pi*np.arange(M)/M)
    used=set(); errs=[]
    for xi in x:                                          # nearest-ideal matching
        j=min((j for j in range(M) if j not in used),key=lambda j:abs(xi-ideal[j]))
        used.add(j); errs.append(abs(xi-ideal[j])/rbar)
    ne=max(errs)
    xm=[mp.mpc(z) for z in x]
    V=mp.matrix(M,M)
    for k in range(M):
        for mu in range(M): V[k,mu]=xm[k]**mu
    Vi=V**-1; tail=mp.mpf(0)
    for m in range(M+1,M+1+periods*M):
        c=Vi*mp.matrix([xm[k]**(m-1) for k in range(M)])
        tail+=sum(abs(c[i]) for i in range(M))
    rng=np.random.default_rng(seed); h=rng.uniform(-1,1,M)
    W=(V.T)**-1*mp.matrix([mp.mpf(v) for v in h])
    match=max(abs(sum(W[k]*xm[k]**mu for k in range(M))-h[mu]) for mu in range(M))
    e2e=sum(abs(sum(W[k]*xm[k]**(m-1) for k in range(M))) for m in range(M+1,M+1+periods*M))
    return dict(re=re_spread,ne=ne,tail=float(tail),rbarM=rbar**M,
                match=float(match),e2e=float(e2e))

def make_figure(path="design_verification.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(9.6,3.7))
    Ms=[4,5,6,7,8,10,12,14]; R=[analyze(design(M)) for M in Ms]
    ax[0].semilogy(Ms,[r["tail"] for r in R],'o-',color='#2166ac',
                   label=r'measured tail $\sum_{m>M}\Vert c_m\Vert_1$')
    ax[0].semilogy(Ms,[4*M*r["rbarM"]+8*r["ne"] for M,r in zip(Ms,R)],'k--',
                   label=r'bound $4M\bar r^{\,M}+8\bar\varepsilon$ (Lemma)')
    ax[0].semilogy(Ms,[4*M*r["rbarM"] for M,r in zip(Ms,R)],':',color='#b2182b',label=r'$4M\bar r^{\,M}$ term')
    ax[0].semilogy(Ms,[8*r["ne"] for r in R],':',color='#1b7837',label=r'$8\bar\varepsilon$ term')
    ax[0].set_xlabel(r'memory depth $M$ ($K=M$)'); ax[0].set_ylabel('extrapolation tail')
    ax[0].set_title('(a) two-term tail law'); ax[0].legend(fontsize=7,frameon=False); ax[0].grid(alpha=.25)
    D0s=[50,200,800,3200,12800,51200]; R2=[analyze(design(10,Delta0_override=D)) for D in D0s]
    ax[1].loglog(D0s,[r["ne"] for r in R2],'s-',color='#1b7837',label=r'node error $\bar\varepsilon$')
    ax[1].loglog(D0s,[r["tail"] for r in R2],'o-',color='#2166ac',label='measured tail')
    ax[1].loglog(D0s,[R2[0]["ne"]*D0s[0]/d for d in D0s],'k--',lw=1,label=r'$\propto 1/\Delta_0$')
    ax[1].set_xlabel(r'$\Delta_0$ (units of $\gamma$)'); ax[1].set_title(r'(b) floor $\propto 1/\Delta_0$ ($M=10$)')
    ax[1].legend(fontsize=7,frameon=False); ax[1].grid(alpha=.25,which='both')
    plt.tight_layout(); plt.savefig(path,dpi=200)

if __name__=="__main__":
    print("%3s %9s %9s %10s %11s %11s %9s"%("M","reSpr","nodeErr","rbar^M","tail","bound","match"))
    for M in [4,6,8,10,12,14]:
        r=analyze(design(M))
        print("%3d %9.1e %9.2e %10.1e %11.3e %11.3e %9.1e %s"%
              (M,r["re"],r["ne"],r["rbarM"],r["tail"],4*M*r["rbarM"]+8*r["ne"],r["match"],
               "OK" if r["tail"]<=4*M*r["rbarM"]+8*r["ne"] else "VIOLATION"))
    make_figure(); print("figure written: design_verification.png")
