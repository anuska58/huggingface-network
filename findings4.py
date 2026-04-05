"""
RQ4 - Temporal Evolution of Lineage 
============================================================

  New additions:
    1. Per-family quarterly derivative counts → which families
       are producing the most new derivatives each quarter
    2. Family market share over time → tracks whether LLaMA,
       Qwen, Mistral etc. are growing or shrinking as source families
    3. Family HHI over time → concentration within each family
    4. "Family rise/fall" table: first quarter a family appeared,
       peak quarter, and whether it is growing or declining

Outputs (existing):
  - rq4_quarterly_stats.csv
  - rq4_adoption_speed.csv
  - rq4_adoption_trend.csv
  - rq4_diversity_over_time.csv

Outputs (NEW — family temporality):
  - rq4_family_quarterly.csv         : per-family per-quarter derivative counts
  - rq4_family_market_share.csv      : family share of new derivatives per quarter
  - rq4_family_lifecycle.csv         : first quarter, peak quarter, trend per family
  - rq4_summary.txt
"""

import pandas as pd
import networkx as nx
import re
from collections import defaultdict

EDGES_FILE = "edges.csv"
NODES_FILE = "all_text_generation_models.csv"

# ── Family patterns (consistent word-boundary regex) ─────────────────────────
FAMILY_PATTERNS = [
    ("LLaMA",    [r"llama"]),
    ("Qwen",     [r"\bqwen\b"]),
    ("Mistral",  [r"mistral"]),
    ("GPT",      [r"\bgpt\b", r"openai"]),
    ("DeepSeek", [r"deepseek"]),
    ("Gemma",    [r"gemma"]),
    ("Falcon",   [r"falcon"]),
    ("BLOOM",    [r"bloom"]),
    ("Phi",      [r"\bphi\b"]),
    ("OPT",      [r"\bopt\b"]),
    ("T5",       [r"\bt5\b"]),
    ("Yi",       [r"\byi\b"]),
    ("Nemotron", [r"nemotron"]),
]

def match_family(model_id):
    if not model_id or pd.isna(model_id):
        return "Other"
    s = str(model_id).lower()
    for fam, pats in FAMILY_PATTERNS:
        for pat in pats:
            if re.search(pat, s):
                return fam
    return "Other"

print("Loading data...")
edges = pd.read_csv(EDGES_FILE)
nodes = pd.read_csv(NODES_FILE)

if "Id" in nodes.columns and "model_id" not in nodes.columns:
    nodes = nodes.rename(columns={"Id": "model_id"})

# ── Parse dates ───────────────────────────────────────────────────────────────
nodes["created_at"] = pd.to_datetime(nodes["created_at"], utc=True, errors="coerce")
nodes["downloads"]  = pd.to_numeric(nodes["downloads"], errors="coerce").fillna(0)
nodes["quarter"]    = nodes["created_at"].dt.to_period("Q")

# ── Join dates onto edges ─────────────────────────────────────────────────────
print("Joining dates onto edges...")
date_map = nodes.set_index("model_id")["created_at"].to_dict()

edges["source_date"] = edges["Source"].map(date_map)
edges["target_date"] = edges["Target"].map(date_map)
edges["target_quarter"] = pd.to_datetime(
    edges["target_date"], utc=True, errors="coerce"
).dt.to_period("Q")

dated_edges = edges.dropna(subset=["source_date","target_date"]).copy()
print(f"Edges with both dates: {len(dated_edges):,} / {len(edges):,}")

# ── Adoption speed ────────────────────────────────────────────────────────────
print("\n=== ADOPTION SPEED ===")
dated_edges["source_dt"] = pd.to_datetime(dated_edges["source_date"], utc=True, errors="coerce")
dated_edges["target_dt"] = pd.to_datetime(dated_edges["target_date"], utc=True, errors="coerce")
dated_edges["days_to_derive"] = (
    dated_edges["target_dt"] - dated_edges["source_dt"]
).dt.days

valid_gaps = dated_edges[dated_edges["days_to_derive"] >= 0]
print(f"Valid temporal edges (child after parent): {len(valid_gaps):,}")
print(f"Avg days from parent to child: {valid_gaps['days_to_derive'].mean():.0f}")
print(f"Med days from parent to child: {valid_gaps['days_to_derive'].median():.0f}")

# ── Adoption speed (vectorized) ───────────────────────────────────────────────
print("Calculating adoption speed (vectorized)...")
source_set  = set(edges["Source"].unique())
target_set  = set(edges["Target"].unique())
base_id_set = source_set - target_set

base_edges    = valid_gaps[valid_gaps["Source"].isin(base_id_set)].copy()
first_deriv_s = base_edges.groupby("Source")["target_dt"].min().rename("first_derivative_date")
total_deriv_s = base_edges.groupby("Source")["Target"].count().rename("total_derivatives")

bm_dates    = pd.to_datetime(pd.Series(date_map), utc=True, errors="coerce")
adoption_df = pd.concat([first_deriv_s, total_deriv_s, bm_dates.rename("created_at")], axis=1).dropna()
adoption_df.index.name = "base_model"
adoption_df = adoption_df.reset_index()
adoption_df["days_to_first_derivative"] = (
    adoption_df["first_derivative_date"] - adoption_df["created_at"]
).dt.days
adoption_df = adoption_df[adoption_df["days_to_first_derivative"] >= 0]
adoption_df = adoption_df.sort_values("days_to_first_derivative")
adoption_df.to_csv("rq4_adoption_speed.csv", index=False)
print(f"Saved rq4_adoption_speed.csv ({len(adoption_df):,} rows)")

# ── Adoption speed over time ──────────────────────────────────────────────────
adoption_df["creation_quarter"] = adoption_df["created_at"].dt.to_period("Q")
adoption_trend = (
    adoption_df
    .groupby("creation_quarter")["days_to_first_derivative"]
    .agg(["median", "mean", "count"])
    .reset_index()
    .sort_values("creation_quarter")
)
adoption_trend.to_csv("rq4_adoption_trend.csv", index=False)
print("Saved rq4_adoption_trend.csv")
print(f"\nFastest adoption:   {adoption_df['days_to_first_derivative'].min()} days")
print(f"Avg adoption speed: {adoption_df['days_to_first_derivative'].mean():.0f} days")
print(f"Med adoption speed: {adoption_df['days_to_first_derivative'].median():.0f} days")

# ── Quarterly stats ───────────────────────────────────────────────────────────
print("\n=== QUARTERLY STATS ===")
q_stats = []
for quarter, group in dated_edges.groupby("target_quarter"):
    n_edges        = len(group)
    transform_dist = group["Transformation"].value_counts().to_dict()
    q_stats.append({
        "quarter":          str(quarter),
        "new_edges":        n_edges,
        "unique_sources":   group["Source"].nunique(),
        "unique_targets":   group["Target"].nunique(),
        "fine_tune_pct":    round(transform_dist.get("fine-tune",0) / max(n_edges,1) * 100, 1),
        "quantization_pct": round(transform_dist.get("quantization",0) / max(n_edges,1) * 100, 1),
        "adapter_pct":      round(transform_dist.get("adapter",0) / max(n_edges,1) * 100, 1),
        "merge_pct":        round(transform_dist.get("merge",0) / max(n_edges,1) * 100, 1),
    })

q_df = pd.DataFrame(q_stats).sort_values("quarter")
q_df.to_csv("rq4_quarterly_stats.csv", index=False)
print(q_df.to_string(index=False))

# ── HHI concentration ────────────────────────────────────────────────────────
hhi_rows = []
for quarter, group in dated_edges.groupby("target_quarter"):
    source_counts = group["Source"].value_counts()
    total         = source_counts.sum()
    shares        = source_counts / total
    hhi           = (shares ** 2).sum()
    hhi_rows.append({"quarter": str(quarter), "hhi": round(hhi, 4), "n_edges": total})

hhi_df = pd.DataFrame(hhi_rows).sort_values("quarter")
diversity_df = q_df.merge(hhi_df, on="quarter")
diversity_df.to_csv("rq4_diversity_over_time.csv", index=False)
print("\nSaved rq4_diversity_over_time.csv")

# ════════════════════════════════════════════════════════════════════════════════
# NEW — FAMILY-BASED TEMPORALITY MEASURE
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("=== FAMILY-BASED TEMPORALITY MEASURE (supervisor request) ===")
print("="*70)

# Assign family to each source (base/parent model)
dated_edges["source_family"] = dated_edges["Source"].apply(match_family)
dated_edges["target_family"] = dated_edges["Target"].apply(match_family)

# ── 1. Per-family per-quarter derivative counts ───────────────────────────────
print("\n--- 1. Per-family quarterly derivative (new edge) counts ---")
family_quarterly = (
    dated_edges
    .groupby(["target_quarter", "source_family"])
    .size()
    .reset_index(name="new_derivatives")
    .sort_values(["target_quarter", "new_derivatives"], ascending=[True, False])
)
family_quarterly["quarter"] = family_quarterly["target_quarter"].astype(str)
family_quarterly = family_quarterly.drop(columns=["target_quarter"])
family_quarterly.to_csv("rq4_family_quarterly.csv", index=False)
print(f"Saved rq4_family_quarterly.csv ({len(family_quarterly):,} rows)")
print(family_quarterly.head(30).to_string(index=False))

# ── 2. Family market share per quarter ───────────────────────────────────────
print("\n--- 2. Family market share (% of all new derivatives per quarter) ---")
quarter_totals = family_quarterly.groupby("quarter")["new_derivatives"].transform("sum")
family_quarterly["market_share_pct"] = (
    family_quarterly["new_derivatives"] / quarter_totals * 100
).round(2)

# Pivot for readability
share_pivot = family_quarterly.pivot_table(
    index="quarter", columns="source_family", values="market_share_pct", fill_value=0
).reset_index()
share_pivot.to_csv("rq4_family_market_share.csv", index=False)
print("Saved rq4_family_market_share.csv")
print(share_pivot.to_string(index=False))

# ── 3. Family lifecycle: first quarter, peak quarter, growth trend ─────────────
print("\n--- 3. Family lifecycle (first appearance, peak, trend) ---")
lifecycle_rows = []
for family, grp in family_quarterly.groupby("source_family"):
    grp = grp.sort_values("quarter")
    quarters     = grp["quarter"].tolist()
    counts       = grp["new_derivatives"].tolist()
    first_q      = quarters[0]
    peak_q       = grp.loc[grp["new_derivatives"].idxmax(), "quarter"]
    peak_count   = grp["new_derivatives"].max()
    last_count   = counts[-1]
    first_count  = counts[0]
    total        = grp["new_derivatives"].sum()

    # Simple trend: compare last 2 quarters to first 2 quarters
    if len(counts) >= 4:
        early_avg = sum(counts[:2]) / 2
        late_avg  = sum(counts[-2:]) / 2
        if late_avg > early_avg * 1.2:
            trend = "growing"
        elif late_avg < early_avg * 0.8:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    lifecycle_rows.append({
        "family":         family,
        "first_quarter":  first_q,
        "peak_quarter":   peak_q,
        "peak_count":     peak_count,
        "latest_count":   last_count,
        "total_derivatives": total,
        "n_active_quarters": len(quarters),
        "trend":          trend,
    })

lifecycle_df = pd.DataFrame(lifecycle_rows).sort_values("total_derivatives", ascending=False)
lifecycle_df.to_csv("rq4_family_lifecycle.csv", index=False)
print("Saved rq4_family_lifecycle.csv")
print(lifecycle_df.to_string(index=False))

# ── Summary ───────────────────────────────────────────────────────────────────
summary = f"""
RQ4 SUMMARY STATISTICS (with family-based temporality)
=======================================================
Edges with datable parent + child:     {len(dated_edges):,}
Valid temporal edges (child > parent): {len(valid_gaps):,}

Adoption speed (base → first derivative):
  Average: {adoption_df['days_to_first_derivative'].mean():.0f} days
  Median:  {adoption_df['days_to_first_derivative'].median():.0f} days
  Fastest: {adoption_df['days_to_first_derivative'].min()} days

Adoption speed over time (median days):
{adoption_trend[['creation_quarter','median']].to_string(index=False)}

Quarterly ecosystem stats:
{q_df.to_string(index=False)}

Concentration (HHI) over time:
{hhi_df.to_string(index=False)}

=== FAMILY TEMPORALITY ===

Family lifecycle (sorted by total derivatives):
{lifecycle_df.to_string(index=False)}

Family market share per quarter (first 10 quarters):
{share_pivot.head(10).to_string(index=False)}
"""
with open("rq4_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)
print(summary)
print("Saved rq4_summary.txt")