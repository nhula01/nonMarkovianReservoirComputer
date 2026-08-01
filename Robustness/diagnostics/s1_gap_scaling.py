#!/usr/bin/env python3
"""s1_gap_scaling.py -- Supplementary Fig. S1 (gap_scaling).

Generator (supplement Sec. S.1/S.2):
    Gamma_g = i*Omega + (gamma/2) v v* + (gamma_g/2) a a*
    Omega   = diag(delta_k), equidistant: delta_k = Delta_0*(k-(K-1)/2)
    v       = flat, ||v|| = 1
    a_k  ~  sin(omega_k * tau / 2), normalized, omega_k = omega_0 + Delta_0*k
"""
import argparse
import numpy as np

def build_gamma_g(K, Delta0, gamma, gamma_g, omega0_tau, Delta0_tau):
    k = np.arange(K)
    delta = Delta0 * (k - (K - 1) / 2.0)
    v = np.ones(K) / np.sqrt(K)
    a = np.sin(0.5 * (omega0_tau + Delta0_tau * k))
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("alpha vanishes: omega_k*tau sits on nodes; change omega0_tau")
    a = a / n
    G = (1j * np.diag(delta)
         + (gamma / 2.0) * np.outer(v, v.conj())
         + (gamma_g / 2.0) * np.outer(a, a.conj()))
    return G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gamma-g", type=float, default=1.0)
    p.add_argument("--Delta0", type=float, default=5.0,
                   help="comb spacing in units of gamma (document if changed)")
    p.add_argument("--omega0-tau", type=float, default=3.14159265358979,
                   help="omega_0*tau (generic, off-node)")
    p.add_argument("--Delta0-tau", type=float, default=0.02,
                   help="Delta_0*tau increment per mode index")
    p.add_argument("--Kmin", type=int, default=4)
    p.add_argument("--Kmax", type=int, default=28,
                   help="default sweep covers the device mode numbers used in the paper (K<=28)")
    p.add_argument("--npts", type=int, default=10)
    p.add_argument("--Kfit", type=int, default=0,
                   help="fit slope only over K >= Kfit (0 = full plotted range)")
    p.add_argument("--out", default="gap_scaling.png")
    a = p.parse_args()

    Ks = np.unique(np.round(np.geomspace(a.Kmin, a.Kmax, a.npts)).astype(int))
    min_re, trace_re, min_pair = [], [], []
    for K in Ks:
        G = build_gamma_g(K, a.Delta0, a.gamma, a.gamma_g, a.omega0_tau, a.Delta0_tau)
        lam = np.linalg.eigvals(G)
        min_re.append(lam.real.min())
        trace_re.append(lam.real.sum())
        d = np.abs(lam[:, None] - lam[None, :]) + np.eye(K) * 1e9
        min_pair.append(d.min())
    min_re, trace_re, min_pair = map(np.array, (min_re, trace_re, min_pair))

    sel = Ks >= a.Kfit
    slope = np.polyfit(np.log(Ks[sel]), np.log(min_re[sel]), 1)[0]
    for K, m in zip(Ks, min_re):
        print(f"  K={K:4d}   min Re[lambda]={m:.5e}   K*minRe={K*m:.5f}")
    tr_target = (a.gamma + a.gamma_g) / 2.0
    tr_dev = np.abs(trace_re - tr_target).max()
    bound_ok = np.all(min_re <= tr_target / Ks + 1e-12)
    dist_ok = np.all(min_pair > 0)

    print(f"parameters: gamma={a.gamma} gamma_g={a.gamma_g} Delta0={a.Delta0} "
          f"omega0_tau={a.omega0_tau} Delta0_tau={a.Delta0_tau} K={Ks.min()}..{Ks.max()}")
    print(f"measured log-log slope of min Re[lambda] vs K : {slope:.4f}   "
          f"(lemma predicts ~ -1; historical figure quoted -1.007 at the "
          f"original comb -- update the quoted value to THIS number if this "
          f"script is committed)")
    print(f"sum Re[lambda] = (gamma+gamma_g)/2 to machine precision: "
          f"max deviation {tr_dev:.3e}  {'PASS' if tr_dev < 1e-10 else 'FAIL'}")
    print(f"trace bound min Re <= (gamma+gamma_g)/2K at every K: "
          f"{'PASS' if bound_ok else 'FAIL'}")
    print(f"pairwise distinct at every finite K (min gap "
          f"{min_pair.min():.3e}): {'PASS' if dist_ok else 'FAIL'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax1.loglog(Ks, min_re, "o-", color="C0", label=r"$\min_k\,\mathrm{Re}[\lambda_k]$")
    ax1.loglog(Ks, tr_target / Ks, "--", color="k",
               label=r"trace bound $(\gamma+\gamma_g)/2K$")
    ax1.set_xlabel(r"$K$"); ax1.set_ylabel(r"rate / $\gamma$")
    ax1.set_title(f"gap closes as 1/K (measured slope {slope:.3f})")
    ax1.legend(frameon=False, fontsize=9)
    ax2.semilogx(Ks, min_pair, "s-", color="C2")
    ax2.set_xlabel(r"$K$")
    ax2.set_ylabel(r"$\min_{j\neq k}|\lambda_j-\lambda_k|\,/\,\gamma$")
    ax2.set_title("eigenvalues pairwise distinct at every finite $K$")
    ax2.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(a.out, dpi=220)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
