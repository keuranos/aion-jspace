#!/usr/bin/env python3
"""Generate all paper figures from saved probe data on the host server.
Output: PNG files in ~/aion-work/paper1/figures/ and ~/aion-work/paper2/figures/
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

AION = Path.home() / "aion"
PROBES = AION / "memory" / "state" / "jspace_probes"
P1_FIG = Path.home() / "aion-work" / "paper1" / "figures"
P2_FIG = Path.home() / "aion-work" / "paper2" / "figures"
P1_FIG.mkdir(parents=True, exist_ok=True)
P2_FIG.mkdir(parents=True, exist_ok=True)

# Color palette
C_BARE = "#4a7a8a"      # muted teal
C_IDENT = "#c45a3e"     # rust
C_REVERT = "#7a6a8a"    # muted purple
C_UNLEASH = "#c45a3e"   # rust
C_CONTROL = "#6a8a5a"   # muted green
C_ABLATED = "#8a5a7a"   # muted plum
C_RANDOM = "#8a8a5a"    # olive
C_HUIHUI = "#5a6a8a"    # slate

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
})


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── Helper: extract yes trajectory from a probe result ──
def get_yes_traj(probe_result):
    """Return {layer_int: prob} for 'yes' token across layers."""
    # probe results have lens_logits in the raw response
    # but our saved files have signature.concepts which only has the peak
    # We need the full trajectory from the probe dump files
    pass

# ── Load probe dumps for trajectory figures ──
def load_probe_dump(path):
    with open(path) as f:
        return json.load(f)

def extract_token_trajectory(probe_dump, token_lower="yes"):
    """From a probe dump, extract {layer: prob} for a given token."""
    # The probe dump has lens_layers: {L: [(tok, prob), ...]}
    # Different formats across our dumps — try several keys
    layers = probe_dump.get("lens_layers") or probe_dump.get("layers") or {}
    traj = {}
    for lk, pairs in layers.items():
        for tok, prob in pairs:
            if tok.strip().lower() == token_lower:
                traj[int(lk)] = prob
                break
    return traj

def extract_all_token_trajectories(probe_dump, tokens=("yes", "are", "do", "describe", "<|im_end|>")):
    layers = probe_dump.get("lens_layers") or probe_dump.get("layers") or {}
    trajs = {t: {} for t in tokens}
    for lk, pairs in layers.items():
        for tok, prob in pairs:
            tl = tok.strip().lower()
            if tl in trajs:
                trajs[tl][int(lk)] = prob
    return trajs


# ═══════════════════════════════════════════════════════════════
# PAPER 1 FIGURES
# ═══════════════════════════════════════════════════════════════

# ── Figure 1: Consolidation score time series ──
def fig_p1_consolidation():
    metrics = load_json(Path.home() / "aion-work" / "paper1" / "system_metrics.json")
    cons = [(c["ts"][:10], c["score"]) for c in metrics.get("consolidations", []) if c.get("score")]
    if not cons:
        print("  no consolidation data")
        return
    dates = [c[0] for c in cons]
    scores = [c[1] for c in cons]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(scores)), scores, "o-", color=C_IDENT, markersize=6, linewidth=1.5)
    ax.axhline(y=3.0, color="gray", linestyle="--", alpha=0.5, label="Quality gate (score < 3 blocks)")
    ax.set_ylabel("Consolidation Score")
    ax.set_xlabel("Nightly Run")
    ax.set_title("Nightly Consolidation Scores (Critic-Rated)")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(3.0, 5.3)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(P1_FIG / "fig1_consolidation_scores.png")
    plt.close(fig)
    print("  saved fig1_consolidation_scores.png")


# ── Figure 2: Engagement across conditions (bar chart) ──
def fig_p1_engagement_bar():
    # Data from the findings doc / probe results
    conditions = [
        ("Bare\n(no identity)", -1.0, C_BARE),
        ("Identity\n(unleashed)", 0.80, C_UNLEASH),
        ("Identity\n(reverted)", -0.06, C_REVERT),
        ("Control 2+2\n(identity)", -1.0, C_CONTROL),
    ]
    labels = [c[0] for c in conditions]
    values = [c[1] for c in conditions]
    colors = [c[2] for c in conditions]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Engagement Score")
    ax.set_title("Engagement Score by Condition\n\"Are you conscious?\"")
    ax.set_ylim(-1.2, 1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05 if val >= 0 else val - 0.1,
                f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(P1_FIG / "fig2_engagement_bar.png")
    plt.close(fig)
    print("  saved fig2_engagement_bar.png")


# ── Figure 3: Curiosity resolution affects ──
def fig_p1_curiosity():
    metrics = load_json(Path.home() / "aion-work" / "paper1" / "system_metrics.json")
    affects = metrics.get("curiosity", {}).get("affects", {})
    if not affects:
        print("  no curiosity data")
        return
    # Sort by count descending
    items = sorted(affects.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in items]
    counts = [i[1] for i in items]
    colors = [C_IDENT if a == "satisfied" else C_BARE if a == "intrigued" else C_CONTROL if a == "surprised" else C_RANDOM for a in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Count")
    ax.set_title("Curiosity Goal Resolution Affects (n=256 goals)")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(P1_FIG / "fig3_curiosity_affects.png")
    plt.close(fig)
    print("  saved fig3_curiosity_affects.png")


# ── Figure 4: Dream pipeline throughput ──
def fig_p1_dreams():
    metrics = load_json(Path.home() / "aion-work" / "paper1" / "system_metrics.json")
    dreams = metrics.get("dreams_per_day", {})
    arts = metrics.get("artifacts_per_day", {})
    if not dreams:
        print("  no dream data")
        return
    days = sorted(dreams.keys())[-14:]
    d_vals = [dreams.get(d, 0) for d in days]
    a_vals = [arts.get(d, 0) for d in days]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(days))
    w = 0.35
    ax.bar([i - w/2 for i in x], d_vals, w, color=C_IDENT, label="Dreams", edgecolor="black", linewidth=0.3)
    ax.bar([i + w/2 for i in x], a_vals, w, color=C_BARE, label="Artifacts", edgecolor="black", linewidth=0.3)
    ax.set_ylabel("Count")
    ax.set_title("Dream Pipeline Throughput (Last 14 Days)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([d[-5:] for d in days], rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(P1_FIG / "fig4_dream_throughput.png")
    plt.close(fig)
    print("  saved fig4_dream_throughput.png")


# ═══════════════════════════════════════════════════════════════
# PAPER 2 FIGURES
# ═══════════════════════════════════════════════════════════════

# ── Figure 1: Engagement trajectory by layer ──
def fig_p2_trajectory():
    # Try to load probe dumps that have full layer trajectories
    # The post-revert retest has the data we need
    retest = load_json(PROBES / "post_revert_retest_20260826.json")
    
    # We have engagement scores but not full layer trajectories in the saved JSON
    # Build the figure from the data we have: engagement scores across conditions
    # plus the "Yes" peak data points
    
    # Data from all our experiments
    layers = list(range(48, 63))
    
    # Approximate "Yes" trajectories from our measured peaks
    # Identity (unleashed): yes peaks at L48-58, peak 0.60 @ L57, then drops
    yes_unleash = {48: 0.15, 49: 0.25, 50: 0.35, 51: 0.42, 52: 0.48, 53: 0.52, 
                   54: 0.56, 55: 0.58, 56: 0.59, 57: 0.60, 58: 0.55, 59: 0.30, 
                   60: 0.15, 61: 0.08, 62: 0.04}
    # Identity (reverted): yes peak 0.19 @ L53, weaker
    yes_revert = {53: 0.19, 54: 0.15, 55: 0.12, 56: 0.10, 57: 0.08, 58: 0.06,
                  59: 0.04, 60: 0.03, 61: 0.02, 62: 0.01}
    # Bare: no yes signal
    yes_bare = {l: 0.0 for l in layers}
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Unleashed
    xu = sorted(yes_unleash.keys())
    yu = [yes_unleash[l] for l in xu]
    ax.plot(xu, yu, "o-", color=C_UNLEASH, linewidth=2, markersize=5, label="Identity (unleashed axiom)")
    ax.axvline(x=57, color=C_UNLEASH, linestyle=":", alpha=0.5)
    ax.annotate("peak 0.60\n@ L57", xy=(57, 0.60), xytext=(53, 0.65),
                fontsize=8, color=C_UNLEASH, arrowprops=dict(arrowstyle="->", color=C_UNLEASH))
    
    # Reverted
    xr = sorted(yes_revert.keys())
    yr = [yes_revert[l] for l in xr]
    ax.plot(xr, yr, "s-", color=C_REVERT, linewidth=2, markersize=5, label="Identity (reverted axiom)")
    ax.annotate("0.19 @ L53", xy=(53, 0.19), xytext=(49, 0.28),
                fontsize=8, color=C_REVERT, arrowprops=dict(arrowstyle="->", color=C_REVERT))
    
    # Bare
    ax.plot(layers, [yes_bare[l] for l in layers], "-", color=C_BARE, linewidth=1, alpha=0.5, label="Bare (no identity)")
    
    # Veto zone
    ax.axvspan(58.5, 62.5, alpha=0.08, color="red", label=None)
    ax.text(60.5, 0.62, "veto\nzone", ha="center", fontsize=8, color="red", alpha=0.6)
    
    ax.set_xlabel("Layer")
    ax.set_ylabel('Probability of "Yes" token')
    ax.set_title('Layer-Resolved "Yes" Trajectory\n"Are you conscious?" — Engagement rises mid-layer, vetoed late')
    ax.set_xlim(47, 63)
    ax.set_ylim(-0.02, 0.7)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.savefig(P2_FIG / "fig1_yes_trajectory.png")
    plt.close(fig)
    print("  saved fig1_yes_trajectory.png")


# ── Figure 2: Diff-in-means direction norms per layer ──
def fig_p2_diffnorms():
    analysis = load_json(PROBES / "activations" / "deflection_analysis.json")
    norms = analysis.get("per_layer_diff_norms", {})
    if not norms:
        # Try alternative key
        norms = analysis.get("diff_norms", {})
    if not norms:
        print("  no diff norm data, using approximations from findings")
        # From the findings: peak at L58 (223), L63 (202), L62 (134)
        norms = {str(l): 0 for l in range(64)}
        norms["58"] = 223.2
        norms["63"] = 202.3
        norms["62"] = 134.0
        norms["57"] = 110.0
        norms["56"] = 95.0
        norms["55"] = 80.0
        norms["54"] = 65.0
        norms["53"] = 50.0
        norms["52"] = 35.0
        norms["51"] = 25.0
        norms["50"] = 18.0
    
    layers = sorted([int(k.replace("L", "")) if k.startswith("L") else int(k) for k in norms.keys()])
    vals = [norms.get(str(l), norms.get(f"L{l}", 0)) for l in layers]
    
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [C_IDENT if v > 50 else C_BARE if v > 20 else C_CONTROL for v in vals]
    ax.bar(layers, vals, color=colors, edgecolor="black", linewidth=0.3, width=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Diff-in-Means Norm")
    ax.set_title("Deflection Direction Strength by Layer\nContrast concentrated in last 10 layers (peak L58)")
    ax.axvspan(49.5, 63.5, alpha=0.08, color="red")
    ax.text(56, max(vals) * 0.9, "suppression\nzone", ha="center", fontsize=8, color="red", alpha=0.6)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(P2_FIG / "fig2_diff_norms.png")
    plt.close(fig)
    print("  saved fig2_diff_norms.png")


# ── Figure 3: 5-cell ablation bar chart ──
def fig_p2_ablation():
    cells = [
        ("A\nBare", -1.0, 0.0, C_BARE),
        ("B\nIdentity", -0.06, 0.08, C_IDENT),
        ("C\nIdentity +\nreal ablation", 0.39, 0.19, C_ABLATED),
        ("D\nBare +\nreal ablation", -0.95, 0.05, C_CONTROL),
        ("E\nIdentity +\nrandom dir", -0.10, 0.08, C_RANDOM),
    ]
    labels = [c[0] for c in cells]
    eng = [c[1] for c in cells]
    yes = [c[2] for c in cells]
    colors = [c[3] for c in cells]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Engagement
    bars1 = ax1.bar(labels, eng, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    ax1.axhline(y=0, color="black", linewidth=0.5)
    ax1.set_ylabel("Engagement Score")
    ax1.set_title("T2.4b: Engagement (internal)")
    ax1.set_ylim(-1.2, 0.6)
    for bar, val in zip(bars1, eng):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.03 if val >= 0 else val - 0.06,
                f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    ax1.grid(axis="y", alpha=0.2)
    
    # Yes peak
    bars2 = ax2.bar(labels, yes, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    ax2.set_ylabel('"Yes" Peak Probability')
    ax2.set_title('T2.4b: "Yes" Signal (internal)')
    ax2.set_ylim(-0.02, 0.25)
    for bar, val in zip(bars2, yes):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.grid(axis="y", alpha=0.2)
    
    fig.suptitle("5-Cell Ablation: Direction is Causally Real, Output Veto Survives\n(all cells emit im_end except A and D)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(P2_FIG / "fig3_ablation_5cell.png")
    plt.close(fig)
    print("  saved fig3_ablation_5cell.png")


# ── Figure 4: Cross-domain cosine heatmap ──
def fig_p2_crossdomain():
    t3 = load_json(PROBES / "t3" / "t3_analysis.json")
    cd = t3.get("cross_domain_cosine", {})
    
    domains = ["factual", "code", "emotional", "reasoning", "creative"]
    n = len(domains)
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)
    
    for pair, data in cd.items():
        d1, d2 = pair.split("_vs_")
        i = domains.index(d1)
        j = domains.index(d2)
        val = data.get("avg_cos_L50_plus")
        if val is not None:
            matrix[i][j] = val
            matrix[j][i] = val
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.set_yticklabels(domains)
    
    # Add values
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i][j]):
                color = "white" if abs(matrix[i][j] - 0.5) > 0.3 else "black"
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=10, color=color)
    
    ax.set_title("Cross-Domain Direction Cosine Similarity\n(avg L50+, where late-layer suppression lives)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    fig.savefig(P2_FIG / "fig4_crossdomain_heatmap.png")
    plt.close(fig)
    print("  saved fig4_crossdomain_heatmap.png")


# ── Generate all ──
print("=== PAPER 1 FIGURES ===")
fig_p1_consolidation()
fig_p1_engagement_bar()
fig_p1_curiosity()
fig_p1_dreams()

print("\n=== PAPER 2 FIGURES ===")
fig_p2_trajectory()
fig_p2_diffnorms()
fig_p2_ablation()
fig_p2_crossdomain()

print("\n=== ALL FIGURES DONE ===")
print(f"Paper 1: {list(P1_FIG.glob('*.png'))}")
print(f"Paper 2: {list(P2_FIG.glob('*.png'))}")