"""
Representative mode-space generator Gamma_g for the emitter-mirror device.

Built from the manuscript's stated structural form:
    Gamma_g = i*Omega + (gamma/2) (v_meas x v_meas*) + (gamma_g/2) (alpha x alpha*)
with
    Omega   = diag(delta_k),  delta_k = Delta0*k + Delta2*k*|k|   (rotating frame at omega_L)
    alpha_k ∝ sin(omega_k tau / 2) = sin((pi - phi)/2 + delta_k*tau/2)
    ||alpha|| = ||v_meas|| = 1

Stated operating parameters (main text + Supplement):
    gamma = 0.1, tau = 10  ->  gamma*tau = 1
    phi   = pi/3
    c0    = 0.36*gamma = 0.036   (delay-stability margin at gamma*tau = 1)
    T     = 10                   (=> c0*T/2 = 0.18, the tail exponent)
    T_P   ~ 21                   (=> Delta0 = 2*pi/T_P ~ 0.30)
    K     <= 14

NOT stated anywhere in the manuscript: Delta2 (the chirp), and r = gamma_R/gamma.
Delta2 is therefore swept below rather than assumed.
"""
import numpy as np

GAMMA = 0.1
TAU = 10.0
PHI = np.pi / 3
T_STEP = 10.0
C0 = 0.36 * GAMMA
TAIL_EXPONENT = C0 * T_STEP / 2  # 0.18
DELTA0 = 2 * np.pi / 21.0
GAMMA_G = 0.1  # not separately stated; varied in the sensitivity check


def comb(K, Delta0=DELTA0, Delta2=0.0):
    """Symmetric comb of detunings, chirped by Delta2*k|k|."""
    k = np.arange(K) - (K - 1) / 2.0
    return Delta0 * k + Delta2 * k * np.abs(k)


def gamma_g_matrix(K, Delta0=DELTA0, Delta2=0.0, gamma=GAMMA, gamma_g=GAMMA_G,
                   tau=TAU, phi=PHI, vmeas_kind="uniform"):
    d = comb(K, Delta0, Delta2)
    Omega = np.diag(d)

    # emitter coupling: standing-wave amplitude at the emitter
    alpha = np.sin((np.pi - phi) / 2 + d * tau / 2).astype(complex)
    if np.linalg.norm(alpha) == 0:
        raise ValueError("degenerate alpha")
    alpha /= np.linalg.norm(alpha)

    # measured channel profile
    if vmeas_kind == "uniform":
        v = np.ones(K, dtype=complex)
    elif vmeas_kind == "sin":
        v = np.sin((np.pi - phi) / 2 + d * tau / 2).astype(complex)
    else:
        raise ValueError(vmeas_kind)
    v /= np.linalg.norm(v)

    G = 1j * Omega
    G += (gamma / 2) * np.outer(v, v.conj())
    G += (gamma_g / 2) * np.outer(alpha, alpha.conj())
    return G, alpha, v


def spectrum(K, **kw):
    G, alpha, v = gamma_g_matrix(K, **kw)
    lam, VR = np.linalg.eig(G)
    VL = np.linalg.inv(VR).conj().T  # rows of inv(VR) are left eigvecs (conjugated)
    return lam, VR, VL, alpha, v


if __name__ == "__main__":
    print("=== Validation against manuscript's stated invariants ===\n")

    # (1) trace constraint: sum Re[lambda] = (gamma+gamma_g)/2, independent of K
    print("(1) Trace constraint  sum Re[lam] = (gamma+gamma_g)/2 = %.6f" %
          ((GAMMA + GAMMA_G) / 2))
    for K in [4, 6, 10, 14, 20, 30]:
        lam, *_ = spectrum(K)
        print("    K=%3d   sum Re[lam] = %.12f" % (K, lam.real.sum()))

    # (2) min Re[lam] closes as 1/K  -> manuscript reports measured slope -1.007
    print("\n(2) Gap closing: log-log slope of min Re[lam] vs K")
    Ks = np.array([6, 8, 10, 14, 20, 28, 40, 56, 80])
    mins = []
    for K in Ks:
        lam, *_ = spectrum(K)
        mins.append(lam.real.min())
    mins = np.array(mins)
    slope = np.polyfit(np.log(Ks), np.log(mins), 1)[0]
    print("    measured slope = %.3f   (manuscript reports -1.007)" % slope)
    for K, m in zip(Ks, mins):
        print("    K=%3d   min Re[lam] = %.3e   trace bound (g+gg)/2K = %.3e"
              % (K, m, (GAMMA + GAMMA_G) / (2 * K)))

    # (3) all Re[lam] > 0
    print("\n(3) Positivity of Re[lam] at each finite K")
    for K in [6, 14, 30]:
        lam, *_ = spectrum(K)
        print("    K=%3d   min Re[lam] = %.3e   (>0: %s)"
              % (K, lam.real.min(), lam.real.min() > 0))
