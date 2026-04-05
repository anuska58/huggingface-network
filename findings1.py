"""
Data Section — Model Genealogy Descriptive Statistics
======================================================
This script produces all descriptive statistics needed for the
Data / Dataset section of the paper, replacing RQ1 as an independent
research question.

Covers:
  A. Graph-level overview (nodes, edges, components)
  B. Node taxonomy: roots, intermediates, leaves, isolated
  C. Family definitions and family-level counts
  D. Structural property comparison (downloads, out-degree)
  E. Creator/organisation breakdown
  F. Transformation type distribution
  G. Edge extraction confidence summary

Outputs:
  - data_section_graph_overview.csv
  - data_section_node_taxonomy.csv
  - data_section_family_stats.csv
  - data_section_structural_comparison.csv
  - data_section_creator_breakdown.csv
  - data_section_transformation_dist.csv
  - data_section_summary.txt   ← paste key numbers into paper
"""

import pandas as pd
import networkx as nx
import re

EDGES_FILE = "edges.csv"
NODES_FILE = "merged_dataset.csv"

# ── Word-boundary family patterns (consistent across all RQ scripts) ──────────
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

def extract_family(model_id):
    s = str(model_id).lower()
    for fam, pats in FAMILY_PATTERNS:
        for pat in pats:
            if re.search(pat, s):
                return fam
    return "Other"

def extract_base_model_from_tags(tags_str):
    if not tags_str or pd.isna(tags_str):
        return None
    matches = re.findall(
        r"base_model:(?:finetune:|quantized:|adapter:|merge:)?([^,\s]+)",
        str(tags_str)
    )
    return matches[0].strip() if matches else None

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
edges = pd.read_csv(EDGES_FILE)
nodes = pd.read_csv(NODES_FILE)

nodes["model_id"]  = nodes["model_id"].astype(str)
edges["Source"]    = edges["Source"].astype(str)
edges["Target"]    = edges["Target"].astype(str)
nodes["downloads"] = pd.to_numeric(nodes["downloads"], errors="coerce")
nodes["created_at"]= pd.to_datetime(nodes["created_at"], utc=True, errors="coerce")

print(f"Raw nodes in dataset: {len(nodes):,}")
print(f"Raw edges:            {len(edges):,}")

# ── Build graph ───────────────────────────────────────────────────────────────
G = nx.from_pandas_edgelist(
    edges, source="Source", target="Target",
    edge_attr=["Transformation", "Confidence"],
    create_using=nx.DiGraph()
)

node_ids  = set(nodes["model_id"])
edge_ids  = set(edges["Source"]).union(set(edges["Target"]))
all_ids   = sorted(node_ids.union(edge_ids))
G.add_nodes_from(all_ids)

edge_only_ids = edge_ids - node_ids

all_nodes = pd.DataFrame({"model_id": all_ids})
all_nodes = all_nodes.merge(nodes, on="model_id", how="left")
all_nodes["in_degree"]  = all_nodes["model_id"].map(lambda x: G.in_degree(x))
all_nodes["out_degree"] = all_nodes["model_id"].map(lambda x: G.out_degree(x))
all_nodes["edge_only"]  = ~all_nodes["model_id"].isin(node_ids)
all_nodes["downloads"]  = pd.to_numeric(all_nodes["downloads"], errors="coerce")

# ── A. Graph-level overview ───────────────────────────────────────────────────
print("\n" + "="*60)
print("A. GRAPH-LEVEL OVERVIEW")
print("="*60)

wcc        = list(nx.weakly_connected_components(G))
scc        = list(nx.strongly_connected_components(G))
largest_wcc = max(len(c) for c in wcc)
singletons  = sum(1 for c in wcc if len(c) == 1)
avg_in      = sum(dict(G.in_degree()).values())  / G.number_of_nodes()
avg_out     = sum(dict(G.out_degree()).values()) / G.number_of_nodes()

overview = {
    "total_nodes_in_graph":         G.number_of_nodes(),
    "total_edges":                   G.number_of_edges(),
    "nodes_in_scraped_dataset":      len(node_ids),
    "edge_only_nodes":               len(edge_only_ids),
    "weakly_connected_components":   len(wcc),
    "largest_wcc_size":              largest_wcc,
    "singleton_components":          singletons,
    "avg_in_degree":                 round(avg_in, 4),
    "avg_out_degree":                round(avg_out, 4),
    "graph_density":                 round(nx.density(G), 8),
}

for k, v in overview.items():
    print(f"  {k}: {v}")

pd.DataFrame([overview]).to_csv("data_section_graph_overview.csv", index=False)
print("Saved data_section_graph_overview.csv")

# ── B. Node taxonomy ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("B. NODE TAXONOMY")
print("="*60)

# Assign roles
def assign_role(row):
    ind, outd = row["in_degree"], row["out_degree"]
    if ind == 0 and outd > 0:  return "root"
    if ind > 0 and outd > 0:   return "intermediate"
    if ind > 0 and outd == 0:  return "leaf"
    return "isolated"

all_nodes["lineage_role"] = all_nodes.apply(assign_role, axis=1)

# Counts — over connected nodes only (exclude isolated for % base)
connected = all_nodes[all_nodes["lineage_role"] != "isolated"]
role_counts = all_nodes["lineage_role"].value_counts()
n_total    = len(all_nodes)
n_connected = len(connected)

taxonomy_rows = []
for role in ["root", "intermediate", "leaf", "isolated"]:
    count  = int(role_counts.get(role, 0))
    pct_all = round(count / n_total * 100, 2)
    pct_conn = round(count / max(n_connected, 1) * 100, 2) if role != "isolated" else None
    taxonomy_rows.append({
        "role":        role,
        "count":       count,
        "pct_of_all":  pct_all,
        "pct_of_connected": pct_conn,
    })

taxonomy_df = pd.DataFrame(taxonomy_rows)
print(taxonomy_df.to_string(index=False))
taxonomy_df.to_csv("data_section_node_taxonomy.csv", index=False)

# Edge-only root breakdown
root_nodes  = all_nodes[all_nodes["lineage_role"] == "root"]
eo_roots    = root_nodes[root_nodes["edge_only"]].shape[0]
obs_roots   = root_nodes[~root_nodes["edge_only"]].shape[0]
print(f"\n  Roots in scraped dataset (observable):   {obs_roots:,}")
print(f"  Roots only seen in edges (unobservable): {eo_roots:,}")
print(f"  NOTE: {eo_roots:,} roots may be deleted/private repos or outside the scrape window")

# ── C. Family definitions and counts ─────────────────────────────────────────
print("\n" + "="*60)
print("C. MODEL FAMILY STATISTICS")
print("="*60)

# Two-hop family resolution (same logic as RQ2/RQ3)
tags_map = nodes.set_index("model_id")["tags"].to_dict()

def resolve_family_2hop(model_id, depth=0, visited=None):
    if visited is None: visited = set()
    if model_id in visited or depth > 2: return "Other"
    visited.add(model_id)
    fam = extract_family(model_id)
    if fam != "Other": return fam
    tags = tags_map.get(model_id, "")
    base = extract_base_model_from_tags(tags)
    if base:
        fam = extract_family(base)
        if fam != "Other": return fam
        if depth < 2:
            return resolve_family_2hop(base, depth+1, visited)
    return "Other"

all_nodes["family"] = all_nodes["model_id"].apply(resolve_family_2hop)
root_nodes = all_nodes[all_nodes["lineage_role"] == "root"].copy()
root_nodes["family"] = root_nodes["model_id"].apply(resolve_family_2hop)

# Per-family stats across all nodes
family_all = all_nodes.groupby("family").agg(
    total_nodes        = ("model_id",    "count"),
    root_models        = ("lineage_role", lambda x: (x=="root").sum()),
    leaf_models        = ("lineage_role", lambda x: (x=="leaf").sum()),
    intermediate_models= ("lineage_role", lambda x: (x=="intermediate").sum()),
    avg_downloads      = ("downloads",   "mean"),
    med_downloads      = ("downloads",   "median"),
    total_downloads    = ("downloads",   "sum"),
    avg_out_degree     = ("out_degree",  "mean"),
    max_out_degree     = ("out_degree",  "max"),
).round(2).sort_values("total_nodes", ascending=False)

# Add: number of direct derivatives per family root (out_degree sum of roots)
root_deriv = root_nodes.groupby("family")["out_degree"].sum().rename("total_direct_derivatives")
family_all = family_all.join(root_deriv, how="left").fillna(0)
family_all["total_direct_derivatives"] = family_all["total_direct_derivatives"].astype(int)

print(family_all.to_string())
family_all.to_csv("data_section_family_stats.csv")
print("\nSaved data_section_family_stats.csv")

# ── D. Structural comparison by role ─────────────────────────────────────────
print("\n" + "="*60)
print("D. STRUCTURAL PROPERTY COMPARISON BY ROLE")
print("="*60)

comparison = all_nodes.groupby("lineage_role").agg(
    count             = ("model_id",    "count"),
    avg_downloads     = ("downloads",   "mean"),
    med_downloads     = ("downloads",   "median"),
    avg_out_degree    = ("out_degree",  "mean"),
    avg_in_degree     = ("in_degree",   "mean"),
    pct_edge_only     = ("edge_only",   "mean"),
).round(2)
comparison["pct_edge_only"] = (comparison["pct_edge_only"] * 100).round(1)

print(comparison.to_string())
comparison.to_csv("data_section_structural_comparison.csv")
print("Saved data_section_structural_comparison.csv")

# ── E. Creator/organisation breakdown ─────────────────────────────────────────
print("\n" + "="*60)
print("E. TOP CREATORS (root models only, observable)")
print("="*60)

obs_root_nodes = root_nodes[~root_nodes["edge_only"]].copy()
creator_stats = obs_root_nodes.groupby("creator", dropna=False).agg(
    root_model_count       = ("model_id",    "count"),
    total_direct_derivatives = ("out_degree","sum"),
    avg_downloads          = ("downloads",   "mean"),
).sort_values("root_model_count", ascending=False)

print(creator_stats.head(20).to_string())
creator_stats.to_csv("data_section_creator_breakdown.csv")
print("Saved data_section_creator_breakdown.csv")

# ── F. Transformation type distribution ──────────────────────────────────────
print("\n" + "="*60)
print("F. TRANSFORMATION TYPE DISTRIBUTION")
print("="*60)

transform_counts = edges["Transformation"].value_counts()
transform_pct    = (transform_counts / len(edges) * 100).round(2)
transform_df = pd.DataFrame({
    "transformation": transform_counts.index,
    "count":          transform_counts.values,
    "pct":            transform_pct.values,
})
print(transform_df.to_string(index=False))
transform_df.to_csv("data_section_transformation_dist.csv", index=False)
print("Saved data_section_transformation_dist.csv")

# ── G. Edge extraction confidence ────────────────────────────────────────────
print("\n" + "="*60)
print("G. EDGE EXTRACTION CONFIDENCE & SOURCE")
print("="*60)

conf_counts  = edges["Confidence"].value_counts().sort_index(ascending=False)
source_counts = edges["ExtractionSource"].value_counts()

print("Confidence distribution:")
for conf, cnt in conf_counts.items():
    print(f"  {conf}: {cnt:,} ({cnt/len(edges)*100:.1f}%)")

print("\nExtraction source:")
for src, cnt in source_counts.items():
    print(f"  {src}: {cnt:,} ({cnt/len(edges)*100:.1f}%)")

conf_df = conf_counts.reset_index()
conf_df.columns = ["confidence", "count"]
conf_df["pct"] = (conf_df["count"] / len(edges) * 100).round(2)
conf_df.to_csv("data_section_confidence_dist.csv", index=False)

# ── Summary printout (paste into paper) ───────────────────────────────────────
n_roots   = int(role_counts.get("root", 0))
n_leaves  = int(role_counts.get("leaf", 0))
n_inter   = int(role_counts.get("intermediate", 0))
n_iso     = int(role_counts.get("isolated", 0))

top_family_by_nodes   = family_all["total_nodes"].idxmax()
top_family_by_derivs  = family_all["total_direct_derivatives"].idxmax()

summary = f"""
DATA SECTION SUMMARY
====================

Graph overview:
  Total nodes (union of dataset + edge references): {G.number_of_nodes():,}
  Total directed edges:                             {G.number_of_edges():,}
  Nodes in scraped HuggingFace dataset:             {len(node_ids):,}
  Edge-only nodes (referenced but not scraped):     {len(edge_only_ids):,}
  Weakly connected components:                      {len(wcc):,}
  Largest weakly connected component:               {largest_wcc:,} nodes
  Singleton (isolated) components:                  {singletons:,}
  Average in-degree:                                {avg_in:.4f}
  Average out-degree:                               {avg_out:.4f}
  Graph density:                                    {nx.density(G):.2e}

Node taxonomy (over all {n_total:,} unique IDs):
  Root models  (in=0, out>0):        {n_roots:,}  ({n_roots/n_total*100:.1f}%)
    - observable (in dataset):       {obs_roots:,}
    - edge-only  (not scraped):      {eo_roots:,}
  Intermediate (in>0, out>0):        {n_inter:,}  ({n_inter/n_total*100:.1f}%)
  Leaf         (in>0, out=0):        {n_leaves:,}  ({n_leaves/n_total*100:.1f}%)
  Isolated     (in=0, out=0):        {n_iso:,}  ({n_iso/n_total*100:.1f}%)

  Connected nodes only ({n_connected:,}):
    Roots:         {n_roots/max(n_connected,1)*100:.1f}%
    Intermediates: {n_inter/max(n_connected,1)*100:.1f}%
    Leaves:        {n_leaves/max(n_connected,1)*100:.1f}%

Model families defined ({len(FAMILY_PATTERNS)} named + Other):
  Largest family by node count: {top_family_by_nodes}
  Largest family by direct derivatives: {top_family_by_derivs}

{family_all[['total_nodes','root_models','total_direct_derivatives','avg_downloads']].to_string()}

Transformation types:
{transform_df.to_string(index=False)}

Edge confidence:
  High-confidence (1.0): {int(conf_counts.get(1.0, 0)):,} ({conf_counts.get(1.0,0)/len(edges)*100:.1f}%)
  Medium (0.9):          {int(conf_counts.get(0.9, 0)):,} ({conf_counts.get(0.9,0)/len(edges)*100:.1f}%)
  Lower (<0.9):          {int(sum(v for k,v in conf_counts.items() if k < 0.9)):,}

Top 15 creators by root model count:
{creator_stats.head(15).to_string()}
"""

with open("data_section_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("\nAll outputs saved. Use data_section_summary.txt for paper numbers.")