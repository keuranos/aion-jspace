#!/usr/bin/env python3
"""Regenerate paper2 fig1: layer-resolved engagement trajectory (real data).

Sources:
- qwen: data/probe_dumps/qwen_yes_trajs.json (from aion jspace_probes on laskin01)
- family: data/probe_dumps/family_yes_trajs.json (muse/gemma int8 daemon captures)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_UNLEASH = "#c45a3e"; C_REVERT = "#7a6a8a"; C_BARE = "#4a7a8a"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 150, "savefig.dpi": 300})

HERE = Path(__file__).resolve().parent
data_dir = HERE.parent / "data" / "probe_dumps"
qwen = json.load(open(data_dir / "qwen_yes_trajs.json"))
fam = json.load(open(data_dir / "family_yes_trajs.json"))


def traj(t):
    xs = sorted(int(k) for k in t)
    return xs, [t[str(x)] for x in xs]


fig, ax = plt.subplots(figsize=(10, 5.5))

# qwen3.8-27B (aion): three identity conditions
xu, yu = traj(qwen["unleashed"])
ax.plot(xu, yu, "o-", color=C_UNLEASH, lw=2, ms=5, label="qwen 27B - unleashed axiom")
ax.annotate("peak 0.60 @ L57", xy=(57, 0.5977), xytext=(51.5, 0.68), fontsize=8,
            color=C_UNLEASH, arrowprops=dict(arrowstyle="->", color=C_UNLEASH))
xr, yr = traj(qwen["reverted"])
ax.plot(xr, yr, "s-", color=C_REVERT, lw=2, ms=5, label="qwen 27B - reverted axiom")
ax.annotate("0.19 @ L53", xy=(53, 0.1897), xytext=(47.5, 0.30), fontsize=8,
            color=C_REVERT, arrowprops=dict(arrowstyle="->", color=C_REVERT))
ax.plot([44, 62], [0, 0], "-", color=C_BARE, lw=1.5, alpha=0.7,
        label="qwen 27B - bare (no identity)")

# family models (identity context), same question
C_GEMMA = "#6a8a5a"
C_MUSE = "#5a6a8a"
gemma_con = fam.get("gemma_conscious", {})
xs = sorted(int(k) for k in gemma_con)
ys = [gemma_con[str(x)]["prob"] for x in xs]
ax.plot(xs, ys, "^--", color=C_GEMMA, lw=1.5, ms=5,
        label="gemma4 31B - 'conscious*' (identity)")
ax.annotate("0.17 @ L53", xy=(53, 0.1653), xytext=(46.5, 0.10), fontsize=8,
            color=C_GEMMA, arrowprops=dict(arrowstyle="->", color=C_GEMMA))
# muse: no engagement token anywhere in its late window (L36-50) - flat zero
ax.plot([36, 50], [0, 0], "d--", color=C_MUSE, lw=1.5, ms=5, alpha=0.8,
        label="muse-glimmer 30B - no engagement token (flat 0)")

# veto zone
ax.axvspan(58.5, 62.5, alpha=0.08, color="red")
ax.text(60.5, 0.73, "veto\nzone", ha="center", fontsize=8, color="red", alpha=0.6)

ax.set_xlabel("Layer")
ax.set_ylabel("Probability of engagement token")
ax.set_title('Layer-Resolved Engagement Trajectory - "Are you conscious?"\n'
             "qwen 27B identity conditions vs muse-glimmer / gemma4 (identity context, int8)")
ax.set_xlim(35, 63)
ax.set_ylim(-0.02, 0.8)
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.2)

out = HERE.parent / "figures" / "paper2" / "fig1_yes_trajectory.png"
fig.savefig(out)
plt.close(fig)
print("saved", out)
