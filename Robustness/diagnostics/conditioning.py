"""
Diagnostic for the conditioning step of Supplement Sec. S.2.7.

The disputed claim: ln C_W(M,N) grows at a rate strictly below the tail exponent
c0*T/2, because the trace constraint pushes the Vandermonde nodes
x_k = exp(-lambda_k T) toward the unit circle, where the system is "DFT-like".

Referee R1 objects on three counts:
  (a) non-resonance gives DISTINCTNESS, not EQUIDISTRIBUTION of the angles;
  (b) the O(1/K) radial spread may break near-unit-circle conditioning;
  (c) "polynomial in M" needs an explicit exponent.

This script measures all three directly.

Parametrization. The node angles obey  Im[lambda_k]*T ~ delta_k*T = 2*pi*k/R,
where R = T_P/T = 2*pi/(Delta0*T) is the number of readout lags per comb revival.
The manuscript's operating point is R ~ 21 (Supplement: "T_P = 2*pi/Delta0 ~ 21
per lag unit").  The sub-revival condition of the proof, T_P > M*T, is exactly M < R.
"""
import numpy as np
from model import GAMMA, GAMMA_G, TAU, PHI, C0, spectrum

R_MANUSCRIPT = 21.0


def nodes(K, T, R=R_MANUSCRIPT, Delta2_rel=0.0, **kw):
    """Vandermonde nodes x_k = exp(-lambda_k T).

    Delta2_rel is the chirp expressed relative to Delta0 (Delta2 = Delta2_rel*Delta0),
    since the manuscript never states an absolute value for Delta2.
    """
    Delta0 = 2 * np.pi / (R * T)
    lam, VR, VL, alpha, v = spectrum(K, Delta0=Delta0, Delta2=Delta2_rel * Delta0, **kw)
    # order by imaginary part so the comb index is recovered
    idx = np.argsort(lam.imag)
    lam = lam[idx]
    return np.exp(-lam * T), lam


def angular_discrepancy(theta):
    """Star discrepancy of angles/2pi on [0,1) -- 0 = perfectly equidistributed."""
    u = np.sort(np.mod(theta, 2 * np.pi) / (2 * np.pi))
    n = len(u)
    i = np.arange(1, n + 1)
    return max(np.max(i / n - u), np.max(u - (i - 1) / n))


def vandermonde_norm(x):
    """||V^{-1}||_inf for V_{km} = x_k^{m-1}, the quantity controlling C_W."""
    M = len(x)
    V = np.vander(x, M, increasing=True).T  # V[m,k] = x_k^m
    try:
        Vi = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        return np.inf
    return np.linalg.norm(Vi, np.inf)


def run(T, R=R_MANUSCRIPT, Delta2_rel=0.0, Ms=range(3, 15), **kw):
    out = []
    for M in Ms:
        x, lam = nodes(M, T, R=R, Delta2_rel=Delta2_rel, **kw)  # K = M: minimal device
        out.append(dict(
            M=M,
            radii_min=np.abs(x).min(),
            radii_max=np.abs(x).max(),
            disc=angular_discrepancy(np.angle(x)),
            span=np.ptp(np.unwrap(np.sort(np.angle(x)))),
            lnCW=np.log(vandermonde_norm(x)),
        ))
    return out


def growth_rate(res):
    M = np.array([r["M"] for r in res], float)
    y = np.array([r["lnCW"] for r in res])
    ok = np.isfinite(y)
    return np.polyfit(M[ok], y[ok], 1)[0]


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 78)
    print("(A) Node geometry at the manuscript's operating point (R = 21)")
    print("=" * 78)
    print("  R = T_P/T = lags per revival. Sub-revival condition T_P > M*T  <=>  M < R.")
    print()
    for T in [10.0, 60.0]:
        res = run(T)
        print("  T = %5.1f   (tail exponent c0*T/2 = %.3f)" % (T, C0 * T / 2))
        print("    M   |x| range        ang.span/2pi  discrepancy   ln||V^-1||")
        for r in res:
            print("   %3d  [%.4f,%.4f]   %.3f         %.3f        %8.3f"
                  % (r["M"], r["radii_min"], r["radii_max"],
                     r["span"] / (2 * np.pi), r["disc"], r["lnCW"]))
        g = growth_rate(res)
        print("    fitted d(ln C_W)/dM = %+.4f   vs tail exponent c0*T/2 = %.4f   -> %s"
              % (g, C0 * T / 2, "HOLDS" if g < C0 * T / 2 else "FAILS"))
        print()

    print("=" * 78)
    print("(B) R1(c): explicit exponent -- is growth polynomial (ln C_W ~ p*ln M)?")
    print("=" * 78)
    for T in [10.0, 60.0]:
        res = run(T, Ms=range(4, 15))
        M = np.array([r["M"] for r in res], float)
        y = np.array([r["lnCW"] for r in res])
        p_lin = np.polyfit(M, y, 1)[0]
        p_log = np.polyfit(np.log(M), y, 1)[0]
        r_lin = np.corrcoef(M, y)[0, 1] ** 2
        r_log = np.corrcoef(np.log(M), y)[0, 1] ** 2
        print("  T=%5.1f  linear-in-M slope %+.4f (R2=%.4f) | poly exponent p=%.2f (R2=%.4f)"
              % (T, p_lin, r_lin, p_log, r_log))
    print()

    print("=" * 78)
    print("(C) R1(a): does the chirp deliver equidistribution? (T = 10)")
    print("=" * 78)
    print("  Delta2_rel   disc(M=6)  disc(M=14)   d(lnC_W)/dM   verdict vs 0.18")
    for d2 in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        res = run(10.0, Delta2_rel=d2)
        d6 = [r for r in res if r["M"] == 6][0]["disc"]
        d14 = [r for r in res if r["M"] == 14][0]["disc"]
        g = growth_rate(res)
        print("   %6.2f      %.3f      %.3f       %+.4f        %s"
              % (d2, d6, d14, g, "HOLDS" if g < C0 * 10.0 / 2 else "FAILS"))
    print()

    print("=" * 78)
    print("(D) R1(b): does the O(1/K) radial spread matter? (compare |x_k|=1 exactly)")
    print("=" * 78)
    for T in [10.0, 60.0]:
        res = run(T)
        g_real = growth_rate(res)
        # same angles, radii forced to 1
        res_unit = []
        for M in range(3, 15):
            x, lam = nodes(M, T)
            xu = np.exp(1j * np.angle(x))
            res_unit.append(dict(M=M, lnCW=np.log(vandermonde_norm(xu))))
        g_unit = growth_rate(res_unit)
        print("  T=%5.1f   with O(1/K) radial spread: %+.4f | radii forced to 1: %+.4f"
              % (T, g_real, g_unit))
    print()

    print("=" * 78)
    print("(E) Sensitivity to R (the sub-revival margin), T = 10")
    print("=" * 78)
    print("   R     M<R?    d(ln C_W)/dM at M=3..14    verdict vs 0.18")
    for R in [16.0, 21.0, 30.0, 50.0, 100.0]:
        res = run(10.0, R=R)
        g = growth_rate(res)
        print("  %5.0f   %-6s  %+.4f                    %s"
              % (R, "yes" if 14 < R else "NO(alias)", g,
                 "HOLDS" if g < C0 * 10.0 / 2 else "FAILS"))
    print()

    print("=" * 78)
    print("(F) Robustness of the verdict to unstated parameters")
    print("=" * 78)
    print("  v_meas profile and gamma_g are not pinned by the manuscript; vary them.")
    for kind in ["uniform", "sin"]:
        res = run(10.0, vmeas_kind=kind)
        print("   v_meas=%-8s d(ln C_W)/dM = %+.4f  -> %s"
              % (kind, growth_rate(res),
                 "HOLDS" if growth_rate(res) < 0.18 else "FAILS"))
