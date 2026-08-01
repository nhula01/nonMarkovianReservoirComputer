"""
Do Assumption 2 (readout conditioning) and Lemma (uniform kernel decay) hold SIMULTANEOUSLY?

Key identity.  The node angular span is
    Theta = (delta_max - delta_min) * T = B * T,
i.e. the occupied band times the readout period.  So the conditioning window
Theta/2pi in [0.91, 1) is a NYQUIST condition: the occupied band must nearly fill,
but not exceed, the readout's Nyquist band 2*pi/T.

Now note  Theta/2pi = K*Delta0*T/2pi = K/R = 1/c  (at K = M), where c = T_P/(M*T)
is the sub-revival margin.  Therefore

    conditioning   requires   1/c >~ 0.91   i.e.  c <~ 1.1
    sub-revival    requires   c > 1

so the memory horizon T_H = M*T sits at T_H/T_P = 1/c in [0.91, 1): PRESSED AGAINST
the revival, with no room to spare.

But Lemma (kerneldecay) is only claimed for T_H < T_P, and its constant C_h is
claimed K-INDEPENDENT there.  Question: does C_h stay bounded as K grows when the
horizon is pushed to T_H/T_P = 0.91-0.95, which is exactly where conditioning forces it?
"""
import numpy as np
from model import C0, gamma_g_matrix

T = 10.0


def Ch(K, R, TH_frac, d2=0.0):
    Delta0 = 2 * np.pi / (R * T)
    T_P = 2 * np.pi / Delta0
    ts = np.linspace(0, TH_frac * T_P, 6000)
    G, alpha, v = gamma_g_matrix(K, Delta0=Delta0, Delta2=d2 * Delta0)
    lam, VR = np.linalg.eig(G)
    a_c = np.linalg.inv(VR) @ alpha
    v_c = np.conj(v) @ VR
    h = np.array([abs(np.sum(v_c * a_c * np.exp(-lam * t))) for t in ts])
    return (h * np.exp(C0 * ts / 2)).max()


if __name__ == "__main__":
    print("Refinement as the theorem prescribes: K = M grows, R = c*K so T_P > M*T.")
    print("Horizon T_H = M*T, so T_H/T_P = 1/c exactly.\n")

    print("=" * 84)
    print("C_h = sup_{t<=T_H} |h_K(t)| exp(+c0 t/2)   as K grows, at each margin c")
    print("=" * 84)
    print("  c      T_H/T_P   conditioning?    C_h at K = 6, 10, 14, 20, 28, 40")
    for c in [3.5, 2.0, 1.5, 1.2, 1.1, 1.05]:
        TH_frac = 1.0 / c
        vals = [Ch(K, c * K, TH_frac) for K in [6, 10, 14, 20, 28, 40]]
        cond = "HOLDS" if c <= 1.1 else "fails"
        print("  %4.2f    %.3f     %-8s  %s"
              % (c, TH_frac, cond, "  ".join("%8.3f" % v for v in vals)))
    print()
    print("  Conditioning (Assumption 2) needs c <~ 1.1  -> the BOTTOM rows.")
    print("  Kernel uniformity (Lemma) needs C_h bounded in K -> read across each row.")
    print()

    print("=" * 84)
    print("Growth of C_h with K, quantified (ratio C_h(K=40)/C_h(K=6))")
    print("=" * 84)
    for c in [3.5, 2.0, 1.5, 1.2, 1.1, 1.05]:
        TH_frac = 1.0 / c
        lo = Ch(6, c * 6, TH_frac)
        hi = Ch(40, c * 40, TH_frac)
        print("  c=%4.2f  (T_H/T_P=%.3f):  C_h  %8.3f -> %10.3f   ratio %9.1fx   %s"
              % (c, TH_frac, lo, hi, hi / lo,
                 "<-- conditioning window" if c <= 1.1 else ""))
    print()
    print("  If the ratio explodes precisely in the conditioning window, the two")
    print("  requirements of the proof cannot be met at the same operating point.")
