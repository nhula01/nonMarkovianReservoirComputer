"""
Asymptotic form of the S.2.7 conditioning claim.

The claim is an M -> infinity statement: C_W(M,N)*exp(-c0*(M-1)*T/2) -> 0.
But the proof ALSO requires the sub-revival condition T_P > M*T, i.e. R > M with
R = T_P/T.  So R cannot be held fixed as M grows: it must grow at least linearly,
R = c*M with c > 1.

Consequence for the node geometry.  The node angles are arg(x_k) = -Im[lam_k]*T
~ -2*pi*k/R, so M nodes occupy an arc of angular width

        2*pi*M/R = 2*pi/c ,

which is CONSTANT along the refinement.  The nodes therefore do NOT fill the circle
as M grows -- they crowd onto a fixed arc.  A Vandermonde on a fixed arc has
||V^{-1}|| growing EXPONENTIALLY in M, not polynomially, with a rate that vanishes
only as the arc closes (c -> 1).

This script measures the exponential rate as a function of c and of the chirp, and
compares it against the tail exponent c0*T/2.

High-precision arithmetic (mpmath) is used because ||V^{-1}|| overflows double
precision well before M = 20 in the clustered regime.
"""
import numpy as np
import mpmath as mp
from model import GAMMA, GAMMA_G, TAU, PHI, C0, spectrum

mp.mp.dps = 60


def nodes_for(M, T, c, Delta2_rel=0.0, vmeas_kind="uniform"):
    """K = M modes, comb refined so that R = T_P/T = c*M (sub-revival margin c)."""
    R = c * M
    Delta0 = 2 * np.pi / (R * T)
    lam, VR, VL, alpha, v = spectrum(M, Delta0=Delta0, Delta2=Delta2_rel * Delta0,
                                     vmeas_kind=vmeas_kind)
    lam = lam[np.argsort(lam.imag)]
    return np.exp(-lam * T)


def lnVinv(x):
    """log ||V^{-1}||_inf in high precision."""
    M = len(x)
    V = mp.matrix(M, M)
    for m in range(M):
        for k in range(M):
            V[m, k] = mp.mpc(x[k]) ** m
    try:
        Vi = V ** -1
    except Exception:
        return float("inf")
    rows = [sum(abs(Vi[i, j]) for j in range(M)) for i in range(M)]
    return float(mp.log(max(rows)))


def rate(T, c, Delta2_rel=0.0, Ms=(6, 8, 10, 12, 14, 16), vmeas_kind="uniform"):
    y = [lnVinv(nodes_for(M, T, c, Delta2_rel, vmeas_kind)) for M in Ms]
    M = np.array(Ms, float)
    y = np.array(y)
    ok = np.isfinite(y)
    slope, icept = np.polyfit(M[ok], y[ok], 1)
    r2 = np.corrcoef(M[ok], y[ok])[0, 1] ** 2
    return slope, r2, y


def arcspan(M, T, c, Delta2_rel=0.0):
    x = nodes_for(M, T, c, Delta2_rel)
    a = np.sort(np.angle(x))
    return np.ptp(a) / (2 * np.pi)


if __name__ == "__main__":
    T = 10.0
    tail = C0 * T / 2
    print("Tail exponent to beat:  c0*T/2 = %.4f   (c0 = 0.36*gamma, T = 10)\n" % tail)

    print("=" * 82)
    print("(1) EQUIDISTANT COMB (Delta2 = 0): rate of ln||V^-1|| vs sub-revival margin c")
    print("=" * 82)
    print("     c = R/M   arc/2pi   d(ln C_W)/dM   R^2      verdict vs %.3f" % tail)
    for c in [1.05, 1.2, 1.5, 2.0, 3.0, 5.0]:
        s, r2, _ = rate(T, c)
        print("     %5.2f     %.3f     %+8.4f    %.4f    %s"
              % (c, arcspan(12, T, c), s, r2, "HOLDS" if s < tail else "FAILS"))
    print("\n  Note: c must exceed 1 for sub-revival (T_P > M*T). The manuscript's")
    print("  operating point has T_P/T ~ 21 against M = 6, i.e. c ~ 3.5.\n")

    print("=" * 82)
    print("(2) CHIRPED COMB: does dispersion widen the arc and rescue conditioning?")
    print("=" * 82)
    print("  (chirp is quoted relative to Delta0; the manuscript states no value)\n")
    print("   c      Delta2_rel   arc/2pi   d(ln C_W)/dM    verdict vs %.3f" % tail)
    for c in [2.0, 3.5]:
        for d2 in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
            s, r2, _ = rate(T, c, d2)
            print("  %4.1f      %5.2f       %.3f     %+8.4f      %s"
                  % (c, d2, arcspan(12, T, c, d2), s, "HOLDS" if s < tail else "FAILS"))
        print()

    print("=" * 82)
    print("(3) The tension, stated as a requirement on c")
    print("=" * 82)
    print("  For each c, the largest tail exponent c0*T/2 the conditioning can beat:")
    for c in [1.05, 1.2, 1.5, 2.0, 3.0, 3.5, 5.0]:
        s, _, _ = rate(T, c)
        need_c0T = 2 * s  # c0*T must exceed 2*slope
        print("    c = %4.2f -> need c0*T/2 > %.4f  (i.e. c0*T > %.3f); have c0*T/2 = %.3f  %s"
              % (c, s, need_c0T, tail, "OK" if s < tail else "VIOLATED"))
