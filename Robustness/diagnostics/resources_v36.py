"""Resource figures of the designed operating point, v36.

Layer O only. Every number here characterises the device; none discharges a
proof obligation. All expressions are derived symbolically in Sec. S.2.5 and
reproduced here so the printed values have a script behind them.

Chain, common to every entry:
    lambda_star = (gamma + gamma_g) / 2K,  K = M
    Lambda_min  = max{ln 2M, (1/M) ln(8 M A / eps)}   -> ln 2M at the minimal period
    Lambda_max  = 2 Lambda_min                        (T_max = 2 T_min, Step 3)
    e0          = gamma / 2                           (rank-two Hermitian part, rho -> 0)
    rho         = ebar_target / (8 s_alpha Lambda_max),  s_alpha <= q^2
    gamma_g     = rho gamma
    B           = K Delta_0 = M Delta_0
    q           = least prime >= 2 K_phys + 1,  K_phys = M
    N           = 1, so beta = 4N+1 = 5 and (4N)^{q-2} = 4^{q-2}
    ebar_target = 1/4        (the eps-free branch, after N5 relaxed it from 1/(4M^{3/2}))

Entries of D1 evaluated:
    fifth  (radial protection): Delta_0 >= 512 N K e0^2 (4N)^{q-2} / gamma_g
                             => B/gamma >= 8192 M^2 q^2 Lambda 4^{q-2}
    third  (defect protection): Delta_0 >= 384 N e0^2 T beta^{M-1} / ebar_target
                             => B/gamma >= 768 M^2 Lambda 5^{M-1}

Condition D6 (coupler flatness): eps_v <= ebar_target / (8 Lambda_max), and the
geometry gives eps_v ~ B/omega_0, so
    omega_0/gamma >= (B/gamma) * 8 Lambda_max / ebar_target
                   = (B/gamma) * 64 Lambda_min          at ebar_target = 1/4
                  >= 524288 M^2 q^2 Lambda^2 4^{q-2}
"""

import math


def least_prime_at_least(n):
    while True:
        if n > 1 and all(n % d for d in range(2, int(n**0.5) + 1)):
            return n
        n += 1


def figures(M, ebar_target=0.25):
    q = least_prime_at_least(2 * M + 1)
    L = math.log(2 * M)                       # Lambda at the minimal readout period
    inv = 1.0 / ebar_target
    reach = 1024 * M**2 * q**2 * (2 * L) * 4**(q - 2) * inv       # fifth entry
    defect = 192 * M**2 * L * 5**(M - 1) * inv                    # third entry
    carrier = reach * 8 * (2 * L) * inv                           # D6
    return dict(q=q, Lambda=L, reach=reach, defect=defect, carrier=carrier)


def retired(M):
    """v35 figures, at the retired budget ebar_target = 1/(4 M^{3/2}).

    Reproduced here only to show what changed; not printed in the manuscript.
    """
    return figures(M, ebar_target=1.0 / (4 * M**1.5))


if __name__ == '__main__':
    print(f"{'M':>2} {'q':>3} {'Lambda':>7} | {'B/gamma (5th)':>14} {'B/gamma (3rd)':>14} "
          f"| {'omega_0/gamma (D6)':>19} | {'D6 at v35 budget':>17}")
    for M in (2, 3, 4):
        f, r = figures(M), retired(M)
        print(f"{M:>2} {f['q']:>3} {f['Lambda']:>7.4f} | {f['reach']:>14.3e} "
              f"{f['defect']:>14.3e} | {f['carrier']:>19.3e} | {r['carrier']:>17.3e}")

    print("\nConsistency checks (must all print OK):")
    f2 = figures(2)
    ok1 = abs(f2['reach'] - 8192 * 4 * 25 * f2['Lambda'] * 64) / f2['reach'] < 1e-12
    print(f"  fifth entry equals 8192 M^2 q^2 Lambda 4^(q-2)          : {'OK' if ok1 else 'FAIL'}")
    ok2 = abs(f2['defect'] - 768 * 4 * f2['Lambda'] * 5) / f2['defect'] < 1e-12
    print(f"  third entry equals 768 M^2 Lambda 5^(M-1)               : {'OK' if ok2 else 'FAIL'}")
    r2 = retired(2)
    ok3 = abs(r2['carrier'] - 5.16e10) / 5.16e10 < 0.02
    print(f"  retired budget reproduces the v35 D6 cost ~5e10 at M=2  : {'OK' if ok3 else 'FAIL'}")
    ok4 = abs(f2['carrier'] / r2['carrier'] - 0.125) < 1e-12
    print(f"  relaxing ebar_target 1/(4M^{{3/2}}) -> 1/4 gains 8x at M=2: {'OK' if ok4 else 'FAIL'}")
