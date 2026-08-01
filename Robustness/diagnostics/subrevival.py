"""
The lemma's ACTUAL claim, tested on its own terms:

    |h_K(t)| <= C_h * exp(-c0*t/2)   for all 0 <= t <= T_H,  T_H < T_P,
    with C_h independent of K.

So the honest test is: sup_{t <= T_H} |h_K(t)| * exp(+c0*t/2), for T_H comfortably
below T_P, as K grows and as the chirp varies.  Peaks AT or BEYOND T_P are not
counterexamples -- the lemma explicitly disclaims those.

Two questions:
  Q1 (K-uniformity)  Is the constant stable as K grows at fixed T_H/T_P?
  Q2 (domination)    Is the undispersed comb really the worst case for C_h,
                     as "we bound the worst, undispersed case" asserts?
"""
import numpy as np
from model import C0, gamma_g_matrix

T_STEP = 10.0
R_MAN = 21.0


def kernel(K, ts, Delta0, Delta2=0.0, **kw):
    G, alpha, v = gamma_g_matrix(K, Delta0=Delta0, Delta2=Delta2, **kw)
    lam, VR = np.linalg.eig(G)
    a_c = np.linalg.inv(VR) @ alpha
    v_c = np.conj(v) @ VR
    return np.array([np.sum(v_c * a_c * np.exp(-lam * t)) for t in ts])


def Ch(K, Delta0, d2, TH_frac):
    T_P = 2 * np.pi / Delta0
    ts = np.linspace(0, TH_frac * T_P, 4000)
    h = np.abs(kernel(K, ts, Delta0, d2 * Delta0))
    return (h * np.exp(C0 * ts / 2)).max()


if __name__ == "__main__":
    T = T_STEP

    print("=" * 84)
    print("Q1: K-uniformity of C_h on a sub-revival window, comb refined so R = T_P/T")
    print("=" * 84)
    print("  Refinement done as the theorem prescribes: T_P grows with K (R = 1.5*K),")
    print("  so the memory horizon stays inside the first revival.")
    print()
    for TH_frac in [0.5, 0.8]:
        print("  T_H = %.1f T_P" % TH_frac)
        print("    K     C_h (d2=0)     C_h (d2 at 80%% of wrap)")
        for K in [6, 10, 14, 20, 28]:
            R = 1.5 * K
            Delta0 = 2 * np.pi / (R * T)
            kmax = (K - 1) / 2
            d2_wrap = (R / (2 * kmax) - 1) / kmax
            print("   %3d    %.4f          %.4f"
                  % (K, Ch(K, Delta0, 0.0, TH_frac),
                     Ch(K, Delta0, 0.8 * d2_wrap, TH_frac)))
        print()

    print("=" * 84)
    print("Q2: is the undispersed comb the worst case for C_h? (K=14, R=21, T_H=0.8 T_P)")
    print("=" * 84)
    K, R = 14, R_MAN
    Delta0 = 2 * np.pi / (R * T)
    kmax = (K - 1) / 2
    d2_wrap = (R / (2 * kmax) - 1) / kmax
    print("  wrap threshold d2 = %.4f" % d2_wrap)
    print()
    print("   d2       arc/2pi    C_h        vs undispersed")
    ref = Ch(K, Delta0, 0.0, 0.8)
    for d2 in [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.094]:
        arc = 2 * kmax * (1 + d2 * kmax) / R
        c = Ch(K, Delta0, d2, 0.8)
        print("  %5.3f      %.3f     %.4f      %5.2fx  %s"
              % (d2, arc, c, c / ref, "<-- WORSE than undispersed" if c > ref * 1.02 else ""))
    print()
    print("  'We bound the worst, undispersed case' requires every row <= 1.00x.")
