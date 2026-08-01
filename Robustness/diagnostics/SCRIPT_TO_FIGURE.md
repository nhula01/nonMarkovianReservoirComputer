- **Fig. S3 (design_verification.png)** <- `complete_route.py`. Verifies the designed operating point D1-D6: D3 equalization (~1e-9), node error ~1/Delta_0, two-term tail law vs the Lemma bound 4M*rbar^M + 8*epsbar, mp-exact matching. Also prints the table quoted in Supplement Sec. S.2.8.
- **Fig. S4 (conditioning.png)** <- `conditioning.py` + `asymptotic.py` -> `makefig.py`. The retained motivation figure ("why weight-norm routes fail"): ln||V^-1|| linear in M on node arcs; rate set by angular span; |x|=1 does not repair it. Uses `model.py` (representative device, validated against manuscript invariants: trace=(gamma+gamma_g)/2 to machine precision; gap slope -1.007).

- `subrevival.py` — K-uniformity of C_h at T_H=0.5*T_P (0.9251->0.9222, K=6-28); chirp-domination refutation data.
- `compat.py` — slow-mode floor / log-horizon evidence (expm-verified; the reason the old Poisson route was abandoned, cf. Remark rem:role).
- `echo.py`, `echo2.py` — revival/echo structure probes.
- `asymptotic.py` — arc-conditioning rate f(c), T-independence (f(1.5)~0.64, f(2.0)~1.06, f(3.5)~1.67).

