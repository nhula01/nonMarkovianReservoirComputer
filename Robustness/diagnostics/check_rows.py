#!/usr/bin/env python3
"""
check_rows.py -- checks the design inequalities of the operating point.

STATUS OF THIS SCRIPT: it checks rows.  It discharges none.  Every inequality it
touches is discharged by a symbolic derivation in the Supplement; this script
only confirms that those derivations were not mis-transcribed.  A PASS
here is not evidence for any lemma.

HONESTY GATE: asserts.  Any violated inequality raises and exits nonzero.
Nothing is tuned to make a check pass.
"""
import itertools, sys
from sympy import isprime, nextprime, Rational, Integer
import mpmath as mp

mp.mp.dps = 60


def smallest_prime_ge(n):
    return n if isprime(n) else nextprime(n)


def admissible_nu(q, K):
    """nu with (2 nu mod q) in {1,...,q-2K+1}."""
    out = []
    for nu in range(q):
        if 1 <= (2 * nu) % q <= q - 2 * K + 1:
            out.append(nu)
    return out


def gen_d(K, N):
    """All d != 0 in Z^K with sum d = 0 and sum |d| <= 2N, generated as
    differences of total-multiplicity vectors of equal order n <= N."""
    seen = set()
    for n in range(1, N + 1):
        vecs = []
        for c in itertools.combinations_with_replacement(range(K), n):
            v = [0] * K
            for i in c:
                v[i] += 1
            vecs.append(tuple(v))
        for a in vecs:
            for b in vecs:
                d = tuple(x - y for x, y in zip(a, b))
                if any(d) and d not in seen:
                    seen.add(d)
    return sorted(seen)


def run(K, N, p=1, verbose=False):
    q = smallest_prime_ge(2 * K + 1)
    assert q <= 4 * K + 2, f"A2 FAIL: Bertrand bound violated, q={q}, K={K}"
    nus = admissible_nu(q, K)
    assert nus, f"A4 FAIL: no admissible nu for q={q}, K={K}"
    nu = nus[0]
    n0 = nu                                   # any n0 congruent to nu works

    # ---- A5: (n0+k)p not divisible by q -------------------------------
    for k in range(K):
        assert ((n0 + k) * p) % q != 0, f"A5 FAIL K={K} k={k}"

    # ---- A6: |sin A_k| >= 2/q -----------------------------------------
    A = [mp.pi * (n0 + k) * p / q for k in range(K)]
    for k in range(K):
        assert abs(mp.sin(A[k])) >= mp.mpf(2) / q - mp.mpf(10) ** -40, f"A6 FAIL K={K} k={k}"

    ds = gen_d(K, N)
    floor12 = mp.mpf(4 * N) ** (-(q - 2))     # (4N)^{-(q-2)}

    worst_R = mp.inf
    for d in ds:
        assert sum(d) == 0 and sum(abs(x) for x in d) <= 2 * N

        # ---- A10: support disjointness -------------------------------
        s1 = {(nu + k) % q for k in range(K) if d[k]}
        s2 = {(-nu - k) % q for k in range(K) if d[k]}
        assert not (s1 & s2), f"A10 FAIL K={K} d={d}"

        # ---- A11/A12: R(zeta) nonzero and above the norm floor -------
        # R(zeta) = 2 Re[ zeta^nu P_d(zeta) ]
        z = mp.e ** (2j * mp.pi * p / q)
        Pd = sum(d[k] * z ** k for k in range(K))
        Rz = 2 * mp.re(z ** nu * Pd)
        assert abs(Rz) > 0, f"A11 FAIL K={K} d={d}"
        assert abs(Rz) >= floor12 * (1 - mp.mpf(10) ** -30), \
            f"A12 FAIL K={K} d={d}: |R|={abs(Rz)} < {floor12}"
        worst_R = min(worst_R, abs(Rz))

    # ---- A7/A9/A14: profile bounds and the radial floor ---------------
    # psi_0 set at its admissible ceiling, the worst case for the lemma.
    psi0 = floor12 / (32 * N)
    beta = 4 * N + 1
    psi = [psi0 * mp.mpf(beta) ** (-k) for k in range(K)]
    at = [mp.sin(A[k] + psi[k]) for k in range(K)]
    for k in range(K):
        assert abs(at[k]) >= mp.mpf(1) / q, f"A7 FAIL K={K} k={k}"
        assert abs(at[k]) <= 1
    D = sum(x ** 2 for x in at)
    assert mp.mpf(K) / q ** 2 <= D <= K, f"A7(D) FAIL K={K}"
    alpha = [x / mp.sqrt(D) for x in at]
    s_alpha = K * max(a ** 2 for a in alpha)
    assert s_alpha <= q ** 2, f"A9 FAIL K={K}: s_alpha={s_alpha} > {q**2}"
    assert abs(min(abs(a) for a in alpha)) >= 1 / (mp.mpf(q) * mp.sqrt(K)), f"A8 FAIL K={K}"

    worst_rad = mp.inf
    target = floor12 / (8 * K)
    for d in ds:
        val = abs(sum(d[k] * alpha[k] ** 2 for k in range(K)))
        assert val >= target * (1 - mp.mpf(10) ** -30), \
            f"A14 FAIL K={K} d={d}: {val} < {target}"
        worst_rad = min(worst_rad, val)

    print(f"  K={K:3d} N={N} q={q:3d} nu={nu:3d} |d|-set={len(ds):6d}  "
          f"min|R|/floor={mp.nstr(worst_R/floor12,4):>10}  "
          f"min|sum d a^2|/floor={mp.nstr(worst_rad/target,4):>10}  "
          f"s_alpha={mp.nstr(s_alpha,4):>8} (<= {q**2})")
    return True


def check_G1():
    """Checks that the OLD (D4) fails to give gamma/4K when N is large
    relative to M, and that the new second entry repairs it."""
    print("\nG1 -- old D4 sufficiency of the gamma/4K claim:")
    bad = []
    for M in (4, 6, 8):
        for N in (2, 5, 20, 60, 200):
            eps = min(1.0 / (8 * M), 1.0)
            s_alpha = (4 * M + 2) ** 2                    # ceiling from A9
            rho_old = eps / (6 * s_alpha * mp.log(2 * M))
            old_ratio = rho_old * N * s_alpha             # want < 1/4
            rho_new = min(rho_old, 1.0 / (8 * N * s_alpha))
            new_ratio = rho_new * N * s_alpha
            ok_old = old_ratio < 0.25
            assert new_ratio <= 0.125 + 1e-12, "G1 FAIL: new entry insufficient"
            if not ok_old:
                bad.append((M, N, float(old_ratio)))
    if bad:
        print("  old entry INSUFFICIENT at:", 
              ", ".join(f"M={m},N={n} -> {r:.3f} (needs <0.25)" for m, n, r in bad))
    else:
        print("  old entry sufficient over the sampled grid (does not prove it in general)")
    print("  new entry rho <= 1/(8 N s_alpha) gives ratio <= 0.125 at every sampled point")


if __name__ == "__main__":
    print("check_rows.py -- CHECKS rows.  Discharges none.\n")
    print("Rows A2,A4,A5,A6,A7,A8,A9,A10,A11,A12,A14:")
    ok = True
    for K, N in [(4, 2), (5, 2), (6, 2), (6, 3), (8, 2), (10, 2)]:
        try:
            run(K, N)
        except AssertionError as e:
            ok = False
            print("  ", e)
    check_G1()
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)
