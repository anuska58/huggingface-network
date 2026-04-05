# paper_figures_clean.py
# Clean, paper-style vector PDFs for RQ2 / RQ3 / RQ4

import os
import math
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import networkx as nx


# ============================================================
# Config
# ============================================================
BASE = r"D:\huggingface_research"   # change to "." if files are in the same folder as this script
OUT = os.path.join(BASE, "paper_figures2_pdf")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "--",
})

# A restrained palette that reads well in PDFs
C1 = "#4E79A7"   # blue
C2 = "#F28E2B"   # orange
C3 = "#59A14F"   # green
C4 = "#E15759"   # red
C5 = "#9D7660"   # brown
C6 = "#76B7B2"   # teal
C7 = "#B07AA1"   # purple
C8 = "#A0A0A0"   # gray

RQ2_FILE = os.path.join(BASE, "rq2_indegree_breakdown_v3.csv")
RQ2_PAIRS = os.path.join(BASE, "rq2_family_pairs_v3.csv")

RQ3_GLOBAL = os.path.join(BASE, "rq3_depth_distribution_global.csv")
RQ3_TOP100 = os.path.join(BASE, "rq3_depth_distribution_top100.csv")
RQ3_BASES = os.path.join(BASE, "rq3_base_model_depths.csv")
EDGES_FILE = os.path.join(BASE, "edges.csv")   # optional, only if you have it locally

RQ4_MIX = os.path.join(BASE, "rq4_diversity_over_time.csv")
RQ4_SPEED = os.path.join(BASE, "rq4_adoption_speed.csv")
RQ4_TREND = os.path.join(BASE, "rq4_adoption_trend.csv")


# ============================================================
# Helpers
# ============================================================
def save_pdf(name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()


def quarter_key(q):
    s = str(q)
    if "Q" not in s or len(s) < 6:
        return (9999, 9)
    return (int(s[:4]), int(s[-1]))


def ordered_quarters(series):
    return sorted(series.astype(str).unique().tolist(), key=quarter_key)


def shorten(model_id: str) -> str:
    s = str(model_id)
    return s.split("/")[-1]


def add_pct_labels(ax, bars, values, threshold=6.0, color="white"):
    for bar, val in zip(bars, values):
        if val < threshold:
            continue
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x, y, f"{val:.0f}%", ha="center", va="center",
                fontsize=9, color=color, fontweight="semibold")


def percentile_stats(frame, group_col, value_col):
    rows = []
    for k, g in frame.groupby(group_col):
        vals = pd.to_numeric(g[value_col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        rows.append({
            group_col: k,
            "count": len(vals),
            "median": float(np.median(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(group_col, key=lambda s: s.map(quarter_key))
    return out


# ============================================================
# RQ2
# ============================================================
def rq2_stacked_by_indegree():
    df = pd.read_csv(RQ2_FILE).sort_values("in_degree")
    x = df["in_degree"].astype(int).to_numpy()

    inb = df["inbreed_pct_of_all"].to_numpy(dtype=float)
    outb = df["outbreed_pct_of_all"].to_numpy(dtype=float)
    unr = df["unresolved_pct_of_all"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    b1 = ax.bar(x, inb, width=0.62, color=C1, label="Inbreeding")
    b2 = ax.bar(x, outb, width=0.62, bottom=inb, color=C2, label="Outbreeding")
    b3 = ax.bar(x, unr, width=0.62, bottom=inb + outb, color=C8, label="Unresolved")

    ax.set_title("Multi-parent composition by in-degree")
    ax.set_xlabel("In-degree")
    ax.set_ylabel("Share of nodes (%)")
    ax.set_xticks(x)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    add_pct_labels(ax, b1, inb, threshold=10)
    add_pct_labels(ax, b2, outb, threshold=3)
    add_pct_labels(ax, b3, unr, threshold=0, color="black")
    save_pdf("rq2_01_indegree_composition_stacked.pdf")


def rq2_top_cross_family_pairs():
    df = pd.read_csv(RQ2_PAIRS).copy()
    df["pair"] = df["family_a"].astype(str) + " × " + df["family_b"].astype(str)
    df = df.sort_values("co_occurrence_count", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.barh(df["pair"], df["co_occurrence_count"], color=C1)
    ax.set_title("Top 10 Most frequent cross-family pairs")
    ax.set_xlabel("Co-occurrence count")
    ax.set_ylabel("")
    ax.xaxis.grid(True, alpha=0.2)
    ax.yaxis.grid(False)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + max(df["co_occurrence_count"]) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{int(w)}",
                va="center", ha="left", fontsize=9)

    save_pdf("rq2_02_cross_family_pairs.pdf")


# ============================================================
# RQ3
# ============================================================
def rq3_depth_cdf():
    g = pd.read_csv(RQ3_GLOBAL).sort_values("depth")
    t = pd.read_csv(RQ3_TOP100).sort_values("depth")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(g["depth"], g["cumulative_pct"], marker="o", linewidth=2.4, color=C1,
            label="Global (all base models)")
    ax.plot(t["depth"], t["cumulative_pct"], marker="o", linewidth=2.0, linestyle="--",
            color=C2, label="Top-100 base models")

    ax.axvline(2, color=C8, linewidth=1.1, linestyle=":")
    ax.axhline(89.3, color=C8, linewidth=1.1, linestyle=":")
    ax.text(2.05, 91.0, "89.3% by depth 2", fontsize=9, color="black")

    ax.set_title("Lineage depth distribution")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Cumulative share (%)")
    ax.set_xticks(g["depth"].astype(int).tolist())
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(frameon=False, loc="lower right")
    save_pdf("rq3_01_depth_cdf.pdf")


def rq3_deepest_lineages_by_family():
    df = pd.read_csv(RQ3_BASES).copy()

    # ── Family mapping (same logic as rq2_family_pairs_v3.csv) ──
    FAMILY_RULES = [
        ("LLaMA",    ["llama", "meta-llama", "l3-", "l3.1-", "l3.2-", "stheno"]),
        ("Mistral",  ["mistral", "mixtral"]),
        ("Qwen",     ["qwen"]),
        ("DeepSeek", ["deepseek"]),
        ("Gemma",    ["gemma"]),
        ("GPT",      ["gpt2", "gpt-", "openai"]),
        ("Nemotron", ["nemotron"]),
        ("Falcon",   ["falcon"]),
        ("Bloom",    ["bloom"]),
        ("Phi",      ["phi-", "/phi"]),
        ("StableLM", ["stablelm"]),
        ("Pythia",   ["pythia"]),
    ]

    def model_to_family(model_id: str) -> str:
        s = str(model_id).lower()
        for family, keywords in FAMILY_RULES:
            if any(kw in s for kw in keywords):
                return family
        return "Other"

    df["family"] = df["model_id"].map(model_to_family)

    # Aggregate: max depth and sum of descendants per family
    agg = df.groupby("family").agg(
        max_depth=("max_depth", "max"),
        total_descendants=("total_descendants", "sum")
    ).reset_index()

    agg = agg.sort_values(
        ["max_depth", "total_descendants"], ascending=[False, False]
    ).head(10).sort_values("max_depth", ascending=True)

    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    bars = ax.barh(agg["family"], agg["max_depth"], color=C1, alpha=0.95)

    ax.set_title("Top 10 Deepest observed lineages (by model family)")
    ax.set_xlabel("Maximum depth")
    ax.set_ylabel("")
    ax.set_xlim(0, max(agg["max_depth"]) + 3)
    ax.xaxis.grid(True, alpha=0.2)
    ax.yaxis.grid(False)

    for bar, total_desc in zip(bars, agg["total_descendants"]):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())}  |  {int(total_desc):,} descendants",
            va="center", ha="left", fontsize=9,
        )

    save_pdf("rq3_02_deepest_lineages_by_family.pdf")

def build_spanning_tree(root, edges_df, max_depth=5, max_nodes=80):
    edges_df = edges_df.copy()
    edges_df.columns = [c.lower() for c in edges_df.columns]

    source_col = "source" if "source" in edges_df.columns else None
    target_col = "target" if "target" in edges_df.columns else None
    if source_col is None or target_col is None:
        raise ValueError("edges.csv must have Source/Target columns (case-insensitive).")

    G = nx.DiGraph()
    for s, t in edges_df[[source_col, target_col]].dropna().itertuples(index=False):
        G.add_edge(str(s), str(t))

    if root not in G:
        raise ValueError(f"Root '{root}' not found in edges graph.")

    tree = nx.DiGraph()
    depth = {root: 0}
    parent = {root: None}
    q = deque([root])
    seen = {root}

    while q and len(seen) < max_nodes:
        u = q.popleft()
        du = depth[u]
        if du >= max_depth:
            continue

        children = sorted(G.successors(u))
        for v in children:
            if v in seen:
                continue
            seen.add(v)
            depth[v] = du + 1
            parent[v] = u
            tree.add_edge(u, v)
            q.append(v)

            if len(seen) >= max_nodes:
                break

    # tidy tree layout
    kids = defaultdict(list)
    for u, v in tree.edges():
        kids[u].append(v)
    for u in kids:
        kids[u] = sorted(kids[u])

    xpos = {}
    next_x = [0]

    def dfs(n):
        children = kids.get(n, [])
        if not children:
            xpos[n] = next_x[0]
            next_x[0] += 1
            return xpos[n]
        child_xs = [dfs(c) for c in children]
        xpos[n] = float(sum(child_xs) / len(child_xs))
        return xpos[n]

    dfs(root)
    pos = {n: (xpos[n], -depth[n]) for n in tree.nodes()}

    return tree, pos, depth


def rq3_optional_tree():
    if not os.path.exists(EDGES_FILE):
        print("edges.csv not found; skipping RQ3 tree figure.")
        return

    bases = pd.read_csv(RQ3_BASES).copy()
    bases = bases.sort_values(["max_depth", "total_descendants"], ascending=[False, False])
    root = str(bases.iloc[0]["model_id"])

    edges = pd.read_csv(EDGES_FILE)
    tree, pos, depth = build_spanning_tree(root, edges, max_depth=5, max_nodes=75)

    nodes = list(tree.nodes())
    node_depths = np.array([depth[n] for n in nodes])
    norm = mpl.colors.Normalize(vmin=node_depths.min(), vmax=max(node_depths.max(), 1))
    cmap = plt.cm.viridis

    sizes = []
    colors = []
    for n in nodes:
        d = depth[n]
        sizes.append(1200 if d == 0 else max(240, 850 - 90 * d))
        colors.append(cmap(norm(d)))

    fig, ax = plt.subplots(figsize=(12.5, 8.8))
    nx.draw_networkx_edges(tree, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=9,
                           width=1.0, alpha=0.28)
    nx.draw_networkx_nodes(tree, pos, ax=ax, node_size=sizes, node_color=colors,
                           edgecolors="black", linewidths=0.5)

    labels = {root: shorten(root)}
    for n in tree.nodes():
        if depth[n] == 1:
            labels[n] = shorten(n)
    nx.draw_networkx_labels(tree, pos, labels=labels, font_size=8, ax=ax)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.01)
    cbar.set_label("Depth")

    ax.set_title(f"Pruned lineage tree for {shorten(root)}")
    ax.axis("off")
    save_pdf("rq3_03_pruned_lineage_tree.pdf")


# ============================================================
# RQ4
# ============================================================
def rq4_transformation_mix():
    df = pd.read_csv(RQ4_MIX).copy()
    df["quarter"] = df["quarter"].astype(str)
    df = df.sort_values("quarter", key=lambda s: s.map(quarter_key))

    known = df[["fine_tune_pct", "quantization_pct", "adapter_pct", "merge_pct"]].sum(axis=1)
    other = (100.0 - known).clip(lower=0)

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12.0, 6.0))

    ax.stackplot(
        x,
        df["fine_tune_pct"],
        df["quantization_pct"],
        df["merge_pct"],
        df["adapter_pct"],
        other,
        labels=["Fine-tune", "Quantization", "Merge", "Adapter", "Other"],
        colors=[C1, C2, C4, C3, C8],
        alpha=0.92
    )

    ax.set_title("Transformation mix over time")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Share of new edges (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["quarter"], rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save_pdf("rq4_01_transformation_mix_area.pdf")


def rq4_adoption_speed_band():
    df = pd.read_csv(RQ4_SPEED).copy()

    if "created_at" not in df.columns or "days_to_first_derivative" not in df.columns:
        raise ValueError("rq4_adoption_speed.csv must contain created_at and days_to_first_derivative columns.")

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_at", "days_to_first_derivative"]).copy()
    df["creation_quarter"] = df["created_at"].dt.to_period("Q").astype(str)

    stats = percentile_stats(df, "creation_quarter", "days_to_first_derivative")
    stats = stats.sort_values("creation_quarter", key=lambda s: s.map(quarter_key))

    x = np.arange(len(stats))
    med = stats["median"].to_numpy()
    q1 = stats["q1"].to_numpy()
    q3 = stats["q3"].to_numpy()

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.fill_between(x, q1, q3, color=C1, alpha=0.18, label="IQR")
    ax.plot(x, med, color=C1, linewidth=2.4, marker="o", label="Median")

    ax.set_title("Adoption speed by creation quarter")
    ax.set_xlabel("Creation quarter")
    ax.set_ylabel("Days to first derivative")
    ax.set_xticks(x)
    ax.set_xticklabels(stats["creation_quarter"], rotation=45, ha="right")

    # Log scale helps with the long-tailed early quarters while keeping the chart readable.
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.legend(frameon=False, loc="upper right")
    save_pdf("rq4_02_adoption_speed_band.pdf")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    rq2_stacked_by_indegree()
    rq2_top_cross_family_pairs()

    rq3_depth_cdf()
    rq3_deepest_lineages_by_family()
    rq3_optional_tree()

    rq4_transformation_mix()
    rq4_adoption_speed_band()

    print(f"Saved PDFs to: {OUT}")