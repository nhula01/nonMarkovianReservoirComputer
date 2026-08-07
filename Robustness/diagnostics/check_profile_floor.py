#!/usr/bin/env python3
"""
check_profile_floor.py -- checks Lemma (Profile normalisation without the delay
lock) of the Supplement.

The lemma states, for an equidistant comb omega_k = omega_0 + Delta_0 k and
theta = Delta_0 tau,

    D = sum_k sin^2(omega_k tau / 2)
      = K/2 - (1/2) cos(omega_0 tau + (K-1) theta/2) * sin(K theta/2)/sin(theta/2)

and consequently D >= K/4 whenever |sin(Delta_0 tau / 2)| >= 2/K.

This script confirms the closed form against the direct sum, and asserts the
floor wherever the stated condition holds.

STATUS: this script CHECKS.  The lemma's proof is symbolic and this script
discharges no step of it.

HONESTY GATE: asserts.  Any violated inequality raises and exits nonzero.
"""
import mpmath as mp

mp.mp.dps = 40


def D_direct(K, w0, D0, tau):
    return sum(mp.sin((w0 + D0 * k) * tau / 2) ** 2 for k in range(K))


def D_closed(K, w0, D0, tau):
    th = D0 * tau
    if abs(mp.sin(th / 2)) < mp.mpf('1e-30'):
        return mp.mpf(K) / 2
    return (mp.mpf(K) / 2
            - mp.cos(w0 * tau + (K - 1) * th / 2)
            * mp.sin(K * th / 2) / mp.sin(th / 2) / 2)


def main():
    print("Check of Lemma (Profile normalisation without the delay lock)")
    print(f"{'K':>4}{'Delta0*tau/2':>14}{'|sin|':>10}{'2/K':>9}{'D':>11}{'K/4':>9}  status")
    w0, tau = mp.mpf('7.31'), mp.mpf(10)
    for K in [6, 10, 14, 20]:
        for half in ['0.05', '0.3', '1.0', '2.0']:
            D0 = mp.mpf(half) * 2 / tau
            direct, closed = D_direct(K, w0, D0, tau), D_closed(K, w0, D0, tau)
            assert abs(direct - closed) < mp.mpf('1e-25'), "closed form disagrees with direct sum"
            sn = abs(mp.sin(D0 * tau / 2))
            holds = sn >= mp.mpf(2) / K
            if holds:
                assert direct >= mp.mpf(K) / 4, f"floor violated at K={K}"
            print(f"{K:4d}{float(D0*tau/2):14.3f}{float(sn):10.4f}{2/K:9.4f}"
                  f"{float(direct):11.4f}{K/4:9.2f}  "
                  f"{'condition holds, D >= K/4' if holds else 'condition not met'}")
    print("PASS: closed form exact, and the floor holds wherever the condition is met.")


if __name__ == "__main__":
    main()
