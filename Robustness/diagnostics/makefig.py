import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import C0
from asymptotic import rate, nodes_for, lnVinv

T = 10.0
tail = C0 * T / 2

fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

# ---- (a) ln||V^-1|| vs M for several sub-revival margins c ----
ax = axes[0]
Ms = np.array([5, 6, 8, 10, 12, 14, 16])
for c, col in zip([1.05, 1.5, 2.0, 3.5], ["#1b7837", "#7fbc41", "#f1a340", "#b2182b"]):
    y = [lnVinv(nodes_for(int(M), T, c)) for M in Ms]
    ax.plot(Ms, y, "o-", color=col, ms=4, lw=1.6,
            label=r"$c=T_P/(MT)=%.2f$" % c)
ax.plot(Ms, tail * (Ms - Ms[0]), "k--", lw=1.5,
        label=r"tail exponent $c_0T/2=%.2f$" % tail)
ax.set_xlabel(r"memory depth $M$ (device refined to $K=M$)")
ax.set_ylabel(r"$\ln\Vert V^{-1}\Vert_\infty$")
ax.set_title(r"(a) growth is exponential in $M$, not polynomial")
ax.legend(fontsize=7.5, frameon=False)
ax.grid(alpha=.25)

# ---- (b) rate vs arc width ----
ax = axes[1]
cs = np.array([1.02, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0])
rates, arcs = [], []
for c in cs:
    s, _, _ = rate(T, c, Ms=(6, 8, 10, 12, 14, 16))
    rates.append(s)
    arcs.append(1.0 / c)
rates = np.array(rates)
arcs = np.array(arcs)
ax.plot(arcs, rates, "o-", color="#2166ac", ms=4.5, lw=1.8)
ax.axhline(tail, color="k", ls="--", lw=1.5)
ax.text(0.10, tail * 1.35, r"tail exponent $c_0T/2=%.2f$" % tail, fontsize=8)
ok = arcs[rates < tail]
if len(ok):
    ax.axvspan(ok.min(), 1.0, color="#1b7837", alpha=.12)
    ax.text(0.72, max(rates) * .62, "admissible\nwindow", fontsize=8,
            ha="center", color="#1b7837")
ax.set_xlabel(r"node angular span / $2\pi$   $\;=\;2k_{\max}(1+\Delta_2 k_{\max}/\Delta_0)/R$")
ax.set_ylabel(r"$d(\ln C_W)/dM$")
ax.set_title(r"(b) conditioning is set by the arc, not the radius")
ax.grid(alpha=.25)
ax.set_ylim(-0.2, max(rates) * 1.12)

# ---- (c) radial vs angular: which mechanism controls? ----
ax = axes[2]
w = 0.36
lab, g_real, g_unit = [], [], []
for c in [1.05, 1.5, 2.0, 3.5]:
    s1, _, _ = rate(T, c, Ms=(6, 8, 10, 12, 14, 16))
    ys = []
    for M in (6, 8, 10, 12, 14, 16):
        x = nodes_for(M, T, c)
        ys.append(lnVinv(np.exp(1j * np.angle(x))))
    s2 = np.polyfit(np.array([6, 8, 10, 12, 14, 16], float), np.array(ys), 1)[0]
    lab.append("%.2f" % c)
    g_real.append(s1)
    g_unit.append(s2)
xp = np.arange(len(lab))
ax.bar(xp - w / 2, g_real, w, label=r"actual nodes ($|x_k|<1$)", color="#4393c3")
ax.bar(xp + w / 2, g_unit, w, label=r"radii forced to $|x_k|=1$", color="#d6604d")
ax.axhline(tail, color="k", ls="--", lw=1.5)
ax.text(len(lab) - 1.5, tail * 1.5, r"$c_0T/2$", fontsize=8)
ax.set_xticks(xp)
ax.set_xticklabels(lab)
ax.set_xlabel(r"sub-revival margin $c=T_P/(MT)$")
ax.set_ylabel(r"$d(\ln C_W)/dM$")
ax.set_title(r"(c) forcing $|x_k|=1$ does not rescue it")
ax.legend(fontsize=7.5, frameon=False)
ax.grid(alpha=.25, axis="y")

plt.tight_layout()
plt.savefig("/home/claude/review/supp/conditioning.png", dpi=200)
print("wrote conditioning.png")
print("\nadmissible arc window (rate < tail):",
      "%.3f -- 1.0" % arcs[rates < tail].min() if (rates < tail).any() else "EMPTY")
print("rate at manuscript operating point c=3.5 (arc=0.29): %.3f  vs tail %.3f"
      % (rates[np.argmin(np.abs(cs - 3.5))], tail))
