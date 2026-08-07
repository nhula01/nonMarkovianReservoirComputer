#!/usr/bin/env python3
"""
check_tail_generic.py -- checks Lemma (Generic-point extrapolation tail).

For distinct nodes x_k with rho = max_k |x_k| < 1/(M+1), the lemma states

    S_out <= M (1+rho)^M sum_{j>=1} rho^j (M+j)^{M+j} / (j^j M^M).

This script builds the mode-space generator of the dispersive comb at generic
operating points (no design condition imposed), forms its nodes x_k =
exp(-lambda_k T), evaluates S_out directly, and asserts the bound.

STATUS: this script CHECKS.  The lemma's proof is symbolic (Hermite contour
formula) and this script discharges no step of it.

HONESTY GATE: asserts.  Any violated inequality raises and exits nonzero.
Nothing is tuned to make a check pass.
"""
import mpmath as mp

mp.mp.dps = 120


def nodes(M, D0, D2, tau, gam, rho_rate, T):
    """Nodes of the dispersive comb at a generic operating point."""
    K = M
    om = [D0 * k + D2 * k * abs(k) for k in range(K)]
    at = [mp.sin(mp.mpf(o) * tau / 2) for o in om]
    nrm = mp.sqrt(sum(a ** 2 for a in at))
    al = [a / nrm for a in at]
    gg = rho_rate * gam
    G = mp.matrix(K, K)
    for i in range(K):
        for j in range(K):
            G[i, j] = (gam / 2) / K + (gg / 2) * al[i] * al[j]
        G[i, i] += mp.mpc(0, 1) * om[i]
    ev = mp.eig(G, left=False, right=False)
    return [mp.e ** (-l * T) for l in ev], ev


def s_out(x, M, mmax):
    K = len(x)
    V = mp.matrix(K, K)
    for k in range(K):
        for mu in range(K):
            V[k, mu] = x[k] ** mu
    Vi = V ** -1
    return sum(sum(abs(ci) for ci in Vi * mp.matrix([x[k] ** (m - 1) for k in range(K)]))
               for m in range(M + 1, mmax + 1))


def bound(rho, M, jmax=60):
    if (M + 1) * rho >= 1:
        return mp.inf
    return M * (1 + rho) ** M * sum(
        rho ** j * mp.mpf(M + j) ** (M + j) / (mp.mpf(j) ** j * mp.mpf(M) ** M)
        for j in range(1, jmax))


def main():
    gam, rho_rate = mp.mpf('0.1'), mp.mpf('0.5')
    D0, D2, tau = mp.mpf('1.0'), mp.mpf('0.05'), mp.mpf(10)
    print("Check of Lemma (Generic-point extrapolation tail)")
    print(f"{'M':>3}{'lam*T':>7}{'rho':>13}{'S_out':>13}{'bound':>12}{'margin':>9}")
    for M in [4, 6, 8]:
        lam = (gam + rho_rate * gam) / (2 * M)
        for f in [5, 8, 12]:
            T = mp.mpf(f) / lam
            x, _ = nodes(M, D0, D2, tau, gam, rho_rate, T)
            rho = max(abs(xi) for xi in x)
            assert (M + 1) * rho < 1, f"hypothesis rho < 1/(M+1) violated at M={M}"
            S, B = s_out(x, M, 12 * M), bound(rho, M)
            assert S <= B, f"BOUND VIOLATED at M={M}, lam*T={f}: {S} > {B}"
            print(f"{M:3d}{f:7d}{mp.nstr(rho,4):>13}{mp.nstr(S,4):>13}"
                  f"{mp.nstr(B,4):>12}{mp.nstr(B/S,4):>9}")
    print("PASS: the bound holds at every generic operating point tested.")


if __name__ == "__main__":
    main()
