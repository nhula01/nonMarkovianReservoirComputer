#!/usr/bin/env python3
"""
check_tail_bound.py -- checks Lemma (Extrapolation tail) of the Supplement.

The lemma states, for nodes x_k = rbar * omega^k * (1 + eps_k) with
|eps_k| <= epsbar <= 1/(4 M^{3/2}), rbar <= 1/(2M) and M >= 2,

    S_out = sum_{m>M} ||c_m||_1  <=  2 M rbar^M + 6 sqrt(M) epsbar .

This script evaluates S_out directly from the node set, at the extreme
admissible values of both parameters and over random phase patterns of eps_k,
and asserts the bound.  It also verifies the two standing hypotheses used in
the proof: ||Y||_{1->1} <= 1/2 and the counting hypothesis S_out <= M.

STATUS: this script CHECKS.  It discharges no step of the lemma, whose proof is
symbolic.  A PASS here is not evidence for the lemma.

HONESTY GATE: asserts.  Any violated inequality raises and exits nonzero.
Nothing is tuned to make a check pass.
"""
import random
import mpmath as mp

mp.mp.dps = 60


def s_out(x, M, mmax=None):
    """S_out computed directly from the Vandermonde system on the node set."""
    K = len(x)
    mmax = mmax or 40 * M
    V = mp.matrix(K, K)
    for k in range(K):
        for mu in range(K):
            V[k, mu] = x[k] ** mu
    Vi = V ** -1
    total = mp.mpf(0)
    for m in range(M + 1, mmax + 1):
        c = Vi * mp.matrix([x[k] ** (m - 1) for k in range(K)])
        term = sum(abs(ci) for ci in c)
        total += term
        if term < mp.mpf('1e-50') and m > M + 2 * K:
            break
    return total


def main(seed=7, trials=8):
    random.seed(seed)
    print("Check of Lemma (Extrapolation tail)")
    print(f"{'M':>4}{'epsbar':>12}{'S_out':>14}{'bound':>14}{'margin':>10}")
    for M in [2, 3, 4, 6, 8, 12]:
        rbar = mp.mpf(1) / (2 * M)
        epsbar = mp.mpf(1) / (4 * M ** mp.mpf(1.5))

        # standing hypothesis of Step 3 of the proof
        Ynorm = 2 * (M - 1) * mp.sqrt(M) * epsbar
        assert Ynorm <= mp.mpf('0.5'), f"||Y|| hypothesis violated at M={M}"

        bound = 2 * M * rbar ** M + 6 * mp.sqrt(M) * epsbar
        assert bound <= M, f"counting hypothesis S_out <= M violated at M={M}"

        worst = mp.mpf(0)
        for _ in range(trials):
            eps = [epsbar * mp.e ** (mp.mpc(0, 1) * mp.mpf(random.uniform(0, 2 * mp.pi)))
                   for _ in range(M)]
            x = [rbar * mp.e ** (-2j * mp.pi * k / M) * (1 + eps[k]) for k in range(M)]
            worst = max(worst, s_out(x, M))

        assert worst <= bound, f"BOUND VIOLATED at M={M}: {worst} > {bound}"
        print(f"{M:4d}{mp.nstr(epsbar, 4):>12}{mp.nstr(worst, 5):>14}"
              f"{mp.nstr(bound, 5):>14}{mp.nstr(bound / worst, 4):>10}")
    print("PASS: the bound holds at every M tested, at the extreme admissible parameters.")


if __name__ == "__main__":
    main()
