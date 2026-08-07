#!/usr/bin/env python3
"""
make_figS8.py -- the three classical sufficient conditions, for the simulated device.

These are the Maass-Legenstein ingredients (separation, fading memory, polynomial
enrichment).  They are an independent sanity check, NOT ingredients of the proof of
Theorem 1, which rests on the Volterra expansion, the resolution of the eigenvalue
sums, the Vandermonde inversion, the extrapolation identity and the tail bound.

Panels:
  (a) separation -- the SIGNED difference of trained outputs for input histories
      differing by delta.  The quantity is signed; the axis is labelled accordingly.
  (b) fading memory -- magnitude of the output difference after a single-step
      perturbation, showing decay to zero.
  (c) polynomial enrichment -- test NRMSE against readout ORDER at FIXED node count.
      Order, not node count, is the variable the property is about.

STATUS: this script CHECKS and PLOTS.  It discharges no step of any lemma.
"""
import numpy as np, itertools, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAM, RHO, TAU, B, CHIRP = 0.1, 5.3, 10.0, 2.0, 0.02
GG = RHO * GAM
TON, TOFF = 1.0, 60.0
MT = 6
LAG = 200                          # device response runs to its natural memory, not the target's
K_SEP, NB_SEP = 14, 30            # panels (a),(b)
K_ENR, NB_ENR = 14, 8             # panel (c): small node set so high orders are feasible

def device(K, NB):
    D0 = B / K; n = np.arange(K)
    w = 100.0 + D0 * n + CHIRP * D0 * n ** 2; d = w - 100.0
    al = np.sin(w * TAU / 2); al /= np.linalg.norm(al)
    v = np.ones(K) / np.sqrt(K)
    G = 1j * np.diag(d) + (GAM / 2) * np.outer(v, v) + (GG / 2) * np.outer(al, al)
    T = TON + TOFF
    lam, VR = np.linalg.eig(G); VL = np.linalg.inv(VR)
    gk = np.array([(v @ VR[:, k]) * (VL[k, :] @ al) for k in range(K)])
    gk = -1j * gk * (1 - np.exp(-lam * TON)) / lam
    xk = np.exp(-lam * T); ts = TOFF * (np.arange(NB) + 0.5) / NB
    E = np.exp(-np.outer(lam, ts))
    return np.array([np.real(gk * xk ** (m - 1) @ E) for m in range(1, LAG + 1)])

def nodes(u, F):
    n = len(u); NB = F.shape[1]; lin = np.zeros((n, NB))
    for m in range(1, LAG + 1): lin[m:] += np.outer(u[:-m], F[m - 1])
    return lin

def poly(lin, order):
    n, NB = lin.shape; cols = [np.ones((n, 1))]
    for deg in range(1, order + 1):
        for c in itertools.combinations_with_replacement(range(NB), deg):
            cols.append(np.prod(lin[:, c], axis=1)[:, None])
    return np.hstack(cols)

def target(u, h1, h2):
    n = len(u); y = np.zeros(n)
    for m in range(1, MT + 1): y[m:] += h1[m - 1] * u[:-m]
    for i in range(1, MT + 1):
        for j in range(i, MT + 1):
            y[max(i, j):] += h2[i - 1, j - 1] * u[max(i, j) - i:n - i] * u[max(i, j) - j:n - j]
    return y

def train(X, y, ntr):
    Xt, yt = X[:ntr], y[:ntr]
    sd = Xt.std(0); sd[sd < 1e-12] = 1.0; mu = Xt.mean(0); mu[0] = 0.0; sd[0] = 1.0
    Xn = (Xt - mu) / sd
    G0 = Xn.T @ Xn
    W = np.linalg.solve(G0 + 1e-8 * np.trace(G0) / G0.shape[0] * np.eye(G0.shape[0]), Xn.T @ yt)
    return lambda Z: ((Z - mu) / sd) @ W          # readout is trained ONCE and held fixed

r = np.random.default_rng(11)
h1 = r.uniform(-1, 1, MT); h2 = np.triu(r.uniform(-1, 1, (MT, MT)))
F = device(K_SEP, NB_SEP)
n = 5000; u = r.uniform(-1, 1, n); y = target(u, h1, h2)
Xu = poly(nodes(u, F), 2); rd = train(Xu, y, 3000); yu = rd(Xu)

deltas = [1e-3, 2e-3, 3e-3]
sep = []
for d in deltas:
    v = u + d * r.uniform(-1, 1, n)
    sep.append(rd(poly(nodes(v, F), 2)) - yu)   # same fixed readout

kick = 1500
fm = []
for d in deltas:
    v = u.copy(); v[kick] += d
    fm.append(np.abs(rd(poly(nodes(v, F), 2)) - yu))

Fe = device(K_ENR, NB_ENR); le = nodes(u, Fe)
orders, errs = [1, 2, 3], []
for o in orders:
    X = poly(le, o); p = train(X, y, 3000)(X)
    errs.append(np.sqrt(np.mean((p[3000:] - y[3000:]) ** 2)) / y[3000:].std())

fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.7))
t = np.arange(3000, 3400)
for d, s in zip(deltas, sep):
    ax[0].plot(t - 3000, s[3000:3400], lw=.9, label=rf'$\delta={d:g}$')
ax[0].axhline(0, color='k', lw=.7)
ax[0].set_xlabel('time step'); ax[0].set_ylabel(r'$\hat y(u,t)-\hat y(v,t)$  (signed)')
ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)
ax[0].set_title('(a)  separation', loc='left', fontweight='bold', fontsize=10)

w = np.arange(kick - 10, kick + 150)
for d, f in zip(deltas, fm):
    ax[1].semilogy(w - kick, np.maximum(f[w], 1e-16), lw=1.1, label=rf'$\delta={d:g}$')
ax[1].axvline(0, color='0.6', ls=':', lw=1.0)
ax[1].set_xlabel('steps since the perturbed input')
ax[1].set_ylabel(r'$|\hat y(u,t)-\hat y(v,t)|$')
ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
ax[1].set_title('(b)  fading memory', loc='left', fontweight='bold', fontsize=10)

ax[2].semilogy(orders, errs, 'o-', c='tab:blue', lw=1.8, ms=6)
for o, e in zip(orders, errs):
    ax[2].annotate(f'{e:.2e}', (o, e), textcoords='offset points', xytext=(0, 8), fontsize=7.5, ha='center')
ax[2].set_xlabel('readout order $n$ (node count fixed at %d)' % NB_ENR)
ax[2].set_ylabel('test NRMSE'); ax[2].set_xticks(orders); ax[2].grid(alpha=.25)
ax[2].set_title('(c)  polynomial enrichment', loc='left', fontweight='bold', fontsize=10)
fig.tight_layout(); fig.savefig('supplementaluniversality.png', dpi=200)

print("separation: max |signed difference| =", [f"{np.max(np.abs(s[3000:])):.3e}" for s in sep])
print("fading memory: value at +40/+80/+120 =", [[f"{f[kick+j]:.2e}" for j in (40,80,120)] for f in fm][0])
print("enrichment: order -> NRMSE")
for o, e in zip(orders, errs): print(f"   n={o}: {e:.4e}")
