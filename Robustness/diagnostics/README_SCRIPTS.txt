  check_localization.py   Lemma (Spectral localization).  Counts eigenvalues per
                          disk by the argument principle applied to the Schur
                          complement, in extended precision (mpmath), and reports
                          which entry of (D1) binds.  Runs at M = 4, 6, 8, 10, 12, 14.
  check_tail_bound.py     Lemma (Extrapolation tail).  Compares the bound
                          2*M*rbar^M + 6*sqrt(M)*epsbar against S_out computed
                          directly from the perturbed node set.
  check_tail_generic.py   Lemma (Generic-point extrapolation tail).  Compares the
                          contour bound against S_out at generic operating points
                          of the dispersive comb.
  check_profile_floor.py  Lemma (Profile normalisation without the delay lock).
                          Confirms the Dirichlet identity and the floor D >= K/4.
  check_rows.py           Design inequalities of the operating point.
  make_figS8.py           Regenerates the universality-properties figure: signed separation,
                          fading-memory decay on a log axis, and enrichment against readout
                          order at fixed node count.  Readout trained once and held fixed.
  make_fig2.py            Regenerates the convergence figure.  The protocol (family rule,
                          fixed node count, envelope definition, seed count and slope
                          metric) is registered in the docstring and was fixed before the
                          sweep was run; the regulariser is selected on validation only.
  make_fig_S1.py          Regenerates the design-verification figure.  Panel (a) computes
                          the extrapolation tail directly from the node set; panel (b)
                          follows the Step-3 selection exactly, carrying Delta_0*T as the
                          exact rational 2*pi*(nM+1)/M so the node phases are exact in
                          double precision, and asserts every entry of (D1) at the
                          realized (T, Delta_0).
  b6_persistence_check.py Persistence baseline on the financial windows.
  b_paired_stats.py       Paired significance protocol (Wilcoxon / McNemar).
  fetch_financial.py      Retrieval of the financial series.
  deposit_manifest.py     SHA-256 manifest of the deposited CSV extracts.


