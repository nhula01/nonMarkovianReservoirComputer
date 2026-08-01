"""
Clean test of the two claims in the uniform-kernel-decay lemma that involve the comb.

CLAIM A ("domination"): "for the mildly dispersive comb used in our numerics the
echoes are additionally chirp-broadened, which only weakens them; we bound the
worst, undispersed case."

CLAIM B ("sub-revival"): |h_K(t)| <= C_h exp(-c0 t/2) for t <= T_H < T_P, with T_P
the UNDISPERSED revival 2*pi/Delta0, "imposed on the undispersed revival time
T_P = 2*pi/Delta0, the conservative choice, since chirp broadens the echoes and only
pushes them later".

Physical reason to doubt both: for a chirped comb the LOCAL mode spacing is
    Delta_local(k) = d(omega)/dk = Delta0 + 2*Delta2*|k| >= Delta0,
so the chirped comb contains spacings LARGER than Delta0, hence partial revivals at
times SHORTER than T_P = 2*pi/Delta0.  Chirp should push echoes EARLIER, not later.

Everything is reported in absolute terms against the envelope the lemma asserts.
"""
import numpy as np
from model import GAMMA, GAMMA_G, C0, gamma_g_matrix, comb

T_STEP = 10.0
R_MAN = 21.0  # T_P / T, from "T_P = 2*pi/Delta0 ~ 21 per lag unit"


def kernel(K, ts, Delta0, Delta2=0.0, **kw):
    G, alpha, v = gamma_g_matrix(K, Delta0=Delta0, Delta2=Delta2, **kw)
    lam, VR = np.linalg.eig(G)
    a_c = np.linalg.inv(VR) @ alpha
    v_c = np.conj(v) @ VR
    return np.array([np.sum(v_c * a_c * np.exp(-lam * t)) for t in ts])


def arc_over_2pi(K, R, d2):
    kmax = (K - 1) / 2.0
    return 2 * kmax * (1 + d2 * kmax) / R


if __name__ == "__main__":
    K = 14
    T = T_STEP
    Delta0 = 2 * np.pi / (R_MAN * T)
    T_P = 2 * np.pi / Delta0
    ts = np.linspace(0, 1.6 * T_P, 8000)
    env = np.exp(-C0 * ts / 2)  # the lemma's asserted shape, up to C_h

    print("K = %d, T = %.0f, Delta0 = %.5f, T_P = %.0f (= %.0f lags)\n"
          % (K, T, Delta0, T_P, T_P / T))

    print("=" * 86)
    print("Angular wrap threshold (nodes must not alias: arc/2pi < 1)")
    print("=" * 86)
    kmax = (K - 1) / 2
    d2_wrap = (R_MAN / (2 * kmax) - 1) / kmax
    print("  arc/2pi = 2*kmax*(1 + d2*kmax)/R  with kmax=%.1f, R=%.0f" % (kmax, R_MAN))
    print("  -> equidistant (d2=0): arc/2pi = %.3f" % arc_over_2pi(K, R_MAN, 0.0))
    print("  -> wraps (arc/2pi = 1) at d2 = %.4f" % d2_wrap)
    print("  Conditioning wants arc/2pi -> 1 (fill circle); aliasing forbids > 1.")
    print()

    print("=" * 86)
    print("CLAIM A/B: peak of |h_K(t)|/C_h_env on (0.2 T_P, 1.5 T_P), vs the envelope")
    print("=" * 86)
    print("  C_h_ratio(t) = |h(t)| / exp(-c0 t/2).  The lemma says this stays bounded")
    print("  by a K-independent constant on t < T_P. A revival shows as a spike.")
    print()
    print("  d2      arc/2pi  wrapped?  max C_h_ratio   at t/T_P   vs d2=0    earlier than T_P?")
    ref = None
    for d2 in [0.0, 0.02, 0.05, 0.09, 0.15, 0.3]:
        h = np.abs(kernel(K, ts, Delta0, d2 * Delta0))
        ratio = h / env
        m = (ts > 0.2 * T_P) & (ts < 1.5 * T_P)
        pk = ratio[m].max()
        tpk = ts[m][np.argmax(ratio[m])] / T_P
        arc = arc_over_2pi(K, R_MAN, d2)
        if ref is None:
            ref = pk
        rel = pk / ref
        print("  %5.2f    %.3f    %-8s  %.4e     %.3f      %6.2fx    %s"
              % (d2, arc, "YES" if arc > 1 else "no", pk, tpk, rel,
                 "YES" if tpk < 0.9 else "no"))
    print()

    print("=" * 86)
    print("Predicted shortest partial-revival time from the largest local spacing")
    print("=" * 86)
    print("  Delta_local_max = Delta0*(1 + 2*d2*kmax)  ->  T_P_eff = T_P/(1 + 2*d2*kmax)")
    print()
    print("  d2      T_P_eff/T_P (predicted)   observed peak t/T_P")
    for d2 in [0.02, 0.05, 0.09, 0.15, 0.3]:
        h = np.abs(kernel(K, ts, Delta0, d2 * Delta0))
        ratio = h / env
        m = (ts > 0.2 * T_P) & (ts < 1.5 * T_P)
        tpk = ts[m][np.argmax(ratio[m])] / T_P
        pred = 1.0 / (1 + 2 * d2 * kmax)
        print("  %5.2f        %.3f                    %.3f" % (d2, pred, tpk))
    print()
    print("  If observed tracks predicted (<1), chirp moves reconstructions EARLIER,")
    print("  contradicting 'chirp ... only pushes them later'.")
