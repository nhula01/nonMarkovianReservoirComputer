#!/usr/bin/env python3
"""fig2_convergence.py -- IMPLEMENTATION of the documented Fig. 2 method.
NOT (yet) the provenance script for the published figure.

WHAT THIS IS: a faithful implementation of the protocol documented in the
manuscript -- main text Sec. 2.2 (convergence protocol: target with known
Volterra kernels M=6, N=2, random bounded coefficients; K-sweep at fixed
mirror distance; linear regression on discretized readout weights; NRMSE
normalized by target range; median/IQR over independent input realizations)
and Methods Sec. 4.8 (displaced-frame quasi-static Bloch route for the fully
saturable overlay: emitter dipole carried as instantaneous steady-state Bloch
response with <sigma_z> = -1/(1+S), field modes evolved under the Gaussian
generator sourced by the dipole; operating saturation depth S_max = 0.2).

This shows an illustration.

USAGE
  demo (runs, banner, no quotable numbers):
      python3 fig2_convergence.py
  candidate provenance run (requires the repo config):
      python3 fig2_convergence.py --repo-config config.json --confirm-repo-config
  config.json keys (all required): Delta0, Delta2, T_on, r, target_seed
  (T_off = 60/gamma is fixed by the manuscript and hard-coded.)
"""
import argparse, json, sys
import numpy as np

GAMMA = 1.0
T_OFF = 60.0 / GAMMA          # stated in main text (throughput paragraph)
M_TARGET, N_TARGET = 6, 2     # stated in main text (Sec. 2.2)
N_SEEDS = 3                   # input realizations (median/IQR), per Fig. 2
LAG_TRUNC = 24                # feature-memory truncation (e^{-Gamma_g T} decay)

DEMO_BANNER = """
############################################################################
# DEMO MODE -- placeholder operating configuration.                         #
# These numbers are NOT the published Fig. 2 and MUST NOT be quoted in or   #
# compared against the manuscript. Supply --repo-config (Delta0, Delta2,    #
# T_on, r, target_seed from the repository pipeline config) together with   #
# --confirm-repo-config to produce a candidate provenance run; then verify  #
# the output against the published curves before committing.                #
############################################################################
"""

# ----------------------------- device -------------------------------------

def build_device(K, cfg):
    k = np.arange(K) - (K - 1) / 2.0
    delta = cfg["Delta0"] * k + cfg["Delta2"] * k**2          # dispersive comb
    v = np.ones(K) / np.sqrt(K)
    a = np.sin(0.5 * (cfg["omega0_tau"] + cfg["Delta0"] * cfg["tau"] * np.arange(K)))
    a = a / np.linalg.norm(a)
    gamma_g = cfg["gamma_g"]
    G = (1j * np.diag(delta) + (GAMMA / 2) * np.outer(v, v)
         + (gamma_g / 2) * np.outer(a, a))
    return G, v, a


def gaussian_features(G, v, a, cfg, u, tgrid):
    """Closed-form linear-transducer features Re[v* e^{-G t} zeta(t_k)]."""
    K = G.shape[0]
    T_on = cfg["T_on"]; T = T_on + T_OFF
    w, R = np.linalg.eig(G); L = np.linalg.inv(R)
    # zeta_m = -i*eta*(1-e^{-G T_on})/G . e^{-G (m-1) T} . alpha   (eta -> 1)
    base = R @ np.diag((1 - np.exp(-w * T_on)) / w) @ L @ a * (-1j)
    zetas = [base.copy()]
    stepT = R @ np.diag(np.exp(-w * T)) @ L
    for _ in range(1, LAG_TRUNC):
        zetas.append(stepT @ zetas[-1])
    zetas = np.array(zetas)                        # (LAG, K)
    prop = np.array([R @ np.diag(np.exp(-w * t)) @ L @ v.conj()
                     for t in tgrid])              # (Ntg, K)  row: v* e^{-G t}
    nsym = len(u)
    F = np.zeros((nsym, len(tgrid)))
    state = np.zeros(zetas.shape[1], complex)
    hist = np.zeros(LAG_TRUNC)
    for kk in range(nsym):
        hist[1:] = hist[:-1]; hist[0] = u[kk]
        zt = hist @ zetas
        F[kk] = (prop @ zt).real
    return F


def bloch_features(G, v, a, cfg, u, tgrid):
    """Displaced-frame quasi-static Bloch route (Methods Sec. 4.8).
    Field modes under the Gaussian generator, sourced by the emitter's
    instantaneous steady-state dipole with <sigma_z> = -1/(1+S)."""
    T_on = cfg["T_on"]; T = T_on + T_OFF
    drive0 = cfg["drive0"]                       # sets S_max at operating drive
    dt = T / 160.0
    nsub = int(round(T / dt))
    non = int(round(T_on / dt))
    z = np.zeros(G.shape[0], complex)
    F = np.zeros((len(u), len(tgrid)))
    tsamp = np.round(np.array(tgrid) / dt).astype(int) + non
    Ssat_track = 0.0
    for kk, uk in enumerate(u):
        for j in range(nsub):
            drv = drive0 * uk if j < non else 0.0
            def rhs(zz):
                loc = drv + (a @ zz)             # instantaneous local drive
                S = 2 * np.abs(loc) ** 2 / GAMMA ** 2
                dip = (-1j * loc / GAMMA) / (1 + S)   # quasi-static <sigma_->
                return -G @ zz + a * dip * GAMMA, S
            k1, S1 = rhs(z)
            k2, _ = rhs(z + 0.5 * dt * k1)
            k3, _ = rhs(z + 0.5 * dt * k2)
            k4, _ = rhs(z + dt * k3)
            z = z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Ssat_track = max(Ssat_track, S1)
            idx = np.where(tsamp == j)[0]
            if idx.size:
                F[kk, idx] = (v.conj() @ z).real
    return F, Ssat_track


# ----------------------------- protocol ------------------------------------

def make_target(seed):
    rng = np.random.default_rng(seed)
    h1 = rng.uniform(-1, 1, M_TARGET)
    h2 = rng.uniform(-1, 1, (M_TARGET, M_TARGET))
    h2 = np.triu(h2) * 0.5
    return h1, h2


def target_output(u, h1, h2):
    n = len(u); M = M_TARGET
    U = np.zeros((n, M))
    for m in range(M):
        U[m:, m] = u[: n - m]
    y = U @ h1 + np.einsum("im,mn,in->i", U, h2, U)
    return y


def readout_nrmse(F_tr, y_tr, F_te, y_te):
    def design(F):
        cols = [np.ones((F.shape[0], 1))]
        for n in range(1, N_TARGET + 1):
            cols.append(F ** n)
        return np.hstack(cols)
    Xtr, Xte = design(F_tr), design(F_te)
    W, *_ = np.linalg.lstsq(Xtr, y_tr, rcond=None)
    err = Xte @ W - y_te
    rng_ = y_te.max() - y_te.min()               # range normalization (Sec. 2.2)
    return np.sqrt(np.mean(err ** 2)) / rng_


def run(cfg, Ks, ntrain, ntest, do_bloch=True):
    h1, h2 = make_target(cfg["target_seed"])
    tgrid = np.linspace(0.5, T_OFF - 0.5, 30)
    res_g = {K: [] for K in Ks}; res_b = {K: [] for K in Ks}
    smax_seen = 0.0
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        u = rng.uniform(-1, 1, ntrain + ntest + M_TARGET)
        y = target_output(u, h1, h2)
        sl_tr = slice(M_TARGET, M_TARGET + ntrain)
        sl_te = slice(M_TARGET + ntrain, None)
        for K in Ks:
            G, v, a = build_device(K, cfg)
            Fg = gaussian_features(G, v, a, cfg, u, tgrid)
            res_g[K].append(readout_nrmse(Fg[sl_tr], y[sl_tr], Fg[sl_te], y[sl_te]))
            if do_bloch:
                Fb, S = bloch_features(G, v, a, cfg, u, tgrid)
                smax_seen = max(smax_seen, S)
                res_b[K].append(readout_nrmse(Fb[sl_tr], y[sl_tr], Fb[sl_te], y[sl_te]))
    return res_g, res_b, smax_seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-config", help="JSON with Delta0, Delta2, T_on, r, target_seed")
    p.add_argument("--confirm-repo-config", action="store_true")
    p.add_argument("--Ks", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10, 12, 14])
    p.add_argument("--ntrain", type=int, default=400)
    p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--no-bloch", action="store_true")
    p.add_argument("--out", default="fig2_convergence.png")
    a = p.parse_args()

    provenance = bool(a.repo_config) and a.confirm_repo_config
    if a.repo_config:
        with open(a.repo_config) as f:
            user_cfg = json.load(f)
        missing = [k for k in ("Delta0", "Delta2", "T_on", "r", "target_seed")
                   if k not in user_cfg]
        if missing:
            sys.exit(f"repo config missing keys: {missing}")
    else:
        user_cfg = dict(Delta0=0.35, Delta2=0.004, T_on=2.0, r=1.0, target_seed=12345)

    cfg = dict(user_cfg)
    cfg.setdefault("gamma_g", 1.0)
    cfg.setdefault("omega0_tau", 3.14159265358979)
    cfg.setdefault("tau", 0.5 / GAMMA)
    cfg.setdefault("drive0", 0.30)   # calibrate so S_max ~= 0.2 at operating drive

    if not provenance:
        print(DEMO_BANNER)
    print("config:", json.dumps(cfg))

    res_g, res_b, smax = run(cfg, a.Ks, a.ntrain, a.ntest, do_bloch=not a.no_bloch)

    def stats(res):
        med = np.array([np.median(res[K]) for K in a.Ks])
        q1 = np.array([np.percentile(res[K], 25) for K in a.Ks])
        q3 = np.array([np.percentile(res[K], 75) for K in a.Ks])
        return med, q1, q3

    mg, g1, g3 = stats(res_g)
    print("\n  K    NRMSE_gauss(median)" + ("   NRMSE_bloch(median)" if not a.no_bloch else ""))
    if not a.no_bloch:
        mb, b1, b3 = stats(res_b)
        for K, x, y_ in zip(a.Ks, mg, mb):
            print(f"  {K:3d}   {x:.5f}              {y_:.5f}")
        print(f"\n  max saturation reached S_max = {smax:.3f}  "
              f"(manuscript operating point: 0.2; adjust drive0 in config)")
        print(f"  Gaussian error reduction across sweep: {mg[0]/mg[-1]:.1f}x ; "
              f"saturable: {mb[0]/mb[-1]:.1f}x (manuscript: seventeen-fold)")
    else:
        for K, x in zip(a.Ks, mg):
            print(f"  {K:3d}   {x:.5f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.semilogy(a.Ks, mg, "o-", color="C0", label="linear-transducer (Gaussian)")
    ax.fill_between(a.Ks, g1, g3, color="C0", alpha=0.2)
    if not a.no_bloch:
        ax.semilogy(a.Ks, mb, "s-", color="C3",
                    label=r"fully saturable (quasi-static Bloch, $S_{\max}\approx$"
                          + f"{smax:.2f})")
        ax.fill_between(a.Ks, b1, b3, color="C3", alpha=0.2)
    ax.axvline(M_TARGET, color="0.5", ls=":", label=r"$K=M$")
    ax.set_xlabel(r"mode number $K$")
    ax.set_ylabel("NRMSE (range-normalized), median/IQR")
    ax.set_title("Convergence of a single device with mode number"
                 + ("  [DEMO CONFIG]" if not provenance else ""))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=220)
    print(f"wrote {a.out}")
    if not provenance:
        print(DEMO_BANNER)


if __name__ == "__main__":
    main()
