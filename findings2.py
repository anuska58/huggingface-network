"""
RQ2 Diagnostic Script for Family Classification
====================================
Goal:
  - Determine whether the family-classification problem comes from the data or from our classifier.
  - Use actual CSV schema:
      edges.csv: Source, Target, Type, Transformation, Confidence, ExtractionSource
      nodes.csv: Id, tasks, creator, model_name, tags, downloads, created_at
  - Compare multiple family-identification methods.

Outputs:
  - rq2_family_method_comparison.csv
  - rq2_resolution_by_method.csv
  - rq2_unresolved_nodes.csv
  - rq2_unresolved_multi_parent_examples.csv
  - rq2_best_effort_family_pairs.csv
  - rq2_diagnostic_summary.txt
"""

import pandas as pd
import networkx as nx
import re
from itertools import combinations
from collections import Counter

EDGES_FILE = "edges.csv"
NODES_FILE = "nodes.csv"

print("Loading data...")
edges = pd.read_csv(EDGES_FILE)
nodes = pd.read_csv(NODES_FILE)

# Normalize column names
if "Id" in nodes.columns and "model_id" not in nodes.columns:
    nodes = nodes.rename(columns={"Id": "model_id"})

# Make sure needed fields exist
for col in ["creator", "model_name", "tags", "downloads", "created_at"]:
    if col not in nodes.columns:
        nodes[col] = "" if col in ["creator", "model_name", "tags", "created_at"] else 0

nodes["downloads"] = pd.to_numeric(nodes["downloads"], errors="coerce").fillna(0)

print(f"Nodes: {len(nodes):,}")
print(f"Edges: {len(edges):,}")

# Build graph
G = nx.from_pandas_edgelist(
    edges,
    source="Source",
    target="Target",
    edge_attr=["Type", "Transformation", "Confidence", "ExtractionSource"],
    create_using=nx.DiGraph()
)

in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())

multi_ids = [n for n, d in in_deg.items() if d >= 2]
multi_nodes = nodes[nodes["model_id"].isin(multi_ids)].copy()

print(f"Multi-parent nodes: {len(multi_nodes):,}")

# -------------------------------------------------------------------
# Family patterns
# Keep this list fairly conservative and paper-friendly.
# -------------------------------------------------------------------
FAMILY_PATTERNS = [
    ("LLaMA",   [r"llama", r"meta-llama"]),
    ("Qwen",    [r"\bqwen\b"]),
    ("Mistral", [r"mistral"]),
    ("GPT",     [r"\bgpt\b", r"openai"]),
    ("DeepSeek",[r"deepseek"]),
    ("Gemma",   [r"gemma"]),
    ("Falcon",  [r"falcon"]),
    ("BLOOM",   [r"bloom"]),
    ("Phi",     [r"\bphi\b"]),
    ("OPT",     [r"\bopt\b"]),
    ("T5",      [r"\bt5\b"]),
    ("Yi",      [r"\byi\b"]),
    ("Nemotron",[r"nemotron"]),
]

def match_family(text):
    if pd.isna(text):
        return None
    s = str(text).lower()
    for fam, pats in FAMILY_PATTERNS:
        for pat in pats:
            if re.search(pat, s):
                return fam
    return None

def extract_base_model_from_tags(tags):
    """
    Pull the base_model field from tags if present.
    Example:
      base_model:Qwen/Qwen3.5-27B
    """
    if pd.isna(tags):
        return None
    s = str(tags)
    m = re.search(r"base_model:([^,]+)", s)
    if m:
        return m.group(1).strip()
    return None

def family_from_base_model_tag(tags):
    base_model = extract_base_model_from_tags(tags)
    return match_family(base_model)

def family_from_model_name(model_name):
    return match_family(model_name)

def family_from_creator(creator):
    return match_family(creator)

def best_effort_family(row):
    """
    Priority:
      1) base_model tag
      2) model_name
      3) creator
    """
    fam = family_from_base_model_tag(row.get("tags", ""))
    if fam is not None:
        return fam, "base_model_tag"

    fam = family_from_model_name(row.get("model_name", ""))
    if fam is not None:
        return fam, "model_name"

    fam = family_from_creator(row.get("creator", ""))
    if fam is not None:
        return fam, "creator"

    return None, None

# -------------------------------------------------------------------
# Assign family labels using multiple methods
# -------------------------------------------------------------------
print("Assigning families using multiple methods...")

nodes["family_tag"] = nodes["tags"].apply(family_from_base_model_tag)
nodes["family_name"] = nodes["model_name"].apply(family_from_model_name)
nodes["family_creator"] = nodes["creator"].apply(family_from_creator)

best_family = []
best_source = []
for _, row in nodes.iterrows():
    fam, src = best_effort_family(row)
    best_family.append(fam)
    best_source.append(src)

nodes["family_best"] = best_family
nodes["family_best_source"] = best_source

# -------------------------------------------------------------------
# Coverage by method
# -------------------------------------------------------------------
coverage_rows = []
for col in ["family_tag", "family_name", "family_creator", "family_best"]:
    coverage_rows.append({
        "method": col,
        "covered_nodes": int(nodes[col].notna().sum()),
        "coverage_pct": round(nodes[col].notna().mean() * 100, 2)
    })

coverage_df = pd.DataFrame(coverage_rows)

print("\n=== FAMILY COVERAGE ===")
print(coverage_df.to_string(index=False))

# Coverage on multi-parent nodes
multi_coverage_rows = []
multi_subset = nodes[nodes["model_id"].isin(multi_ids)].copy()

for col in ["family_tag", "family_name", "family_creator", "family_best"]:
    multi_coverage_rows.append({
        "method": col,
        "covered_multi_parent_nodes": int(multi_subset[col].notna().sum()),
        "coverage_pct": round(multi_subset[col].notna().mean() * 100, 2)
    })

multi_coverage_df = pd.DataFrame(multi_coverage_rows)

print("\n=== MULTI-PARENT COVERAGE ===")
print(multi_coverage_df.to_string(index=False))

# -------------------------------------------------------------------
# Resolution status for each family method
# -------------------------------------------------------------------
def family_status_for_node(node_id, family_col):
    parents = list(G.predecessors(node_id))
    fams = []
    for p in parents:
        val = nodes.loc[nodes["model_id"] == p, family_col]
        if len(val) > 0 and pd.notna(val.iloc[0]):
            fams.append(val.iloc[0])

    fams = sorted(set(fams))
    if len(fams) == 0:
        return "unresolved"
    if len(fams) == 1:
        return "same_family"
    return "cross_family"

resolution_rows = []
for col in ["family_tag", "family_name", "family_creator", "family_best"]:
    counts = Counter()
    for node_id in multi_ids:
        counts[family_status_for_node(node_id, col)] += 1

    total = len(multi_ids)
    same = counts["same_family"]
    cross = counts["cross_family"]
    unresolved = counts["unresolved"]

    resolution_rows.append({
        "method": col,
        "multi_parent_nodes": total,
        "same_family": same,
        "cross_family": cross,
        "unresolved": unresolved,
        "same_family_pct_of_multi": round(same / total * 100, 2),
        "cross_family_pct_of_multi": round(cross / total * 100, 2),
        "unresolved_pct_of_multi": round(unresolved / total * 100, 2),
        "same_family_pct_of_resolved": round(same / max(same + cross, 1) * 100, 2),
        "cross_family_pct_of_resolved": round(cross / max(same + cross, 1) * 100, 2),
    })

resolution_df = pd.DataFrame(resolution_rows)

print("\n=== SAME / CROSS / UNRESOLVED ===")
print(resolution_df.to_string(index=False))

# -------------------------------------------------------------------
# Save unresolved multi-parent examples for manual inspection
# -------------------------------------------------------------------
unresolved_examples = []
for node_id in multi_ids:
    if family_status_for_node(node_id, "family_best") != "unresolved":
        continue

    parents = list(G.predecessors(node_id))
    parent_rows = nodes[nodes["model_id"].isin(parents)][
        ["model_id", "model_name", "creator", "tags", "family_tag", "family_name", "family_creator", "family_best"]
    ].copy()

    unresolved_examples.append({
        "model_id": node_id,
        "creator": nodes.loc[nodes["model_id"] == node_id, "creator"].iloc[0] if len(nodes.loc[nodes["model_id"] == node_id]) > 0 else "",
        "downloads": int(nodes.loc[nodes["model_id"] == node_id, "downloads"].iloc[0]) if len(nodes.loc[nodes["model_id"] == node_id]) > 0 else 0,
        "in_degree": int(in_deg.get(node_id, 0)),
        "parent_count": len(parents),
        "parents": "|".join(parents[:25]),
        "parent_name_matches": int(parent_rows["family_name"].notna().sum()),
        "parent_tag_matches": int(parent_rows["family_tag"].notna().sum()),
        "parent_creator_matches": int(parent_rows["family_creator"].notna().sum()),
        "parent_best_matches": int(parent_rows["family_best"].notna().sum()),
    })

unresolved_df = pd.DataFrame(unresolved_examples).sort_values(
    ["in_degree", "downloads"], ascending=[False, False]
)
unresolved_df.to_csv("rq2_unresolved_multi_parent_examples.csv", index=False)

# Also save unresolved nodes overall
overall_unresolved = nodes[
    nodes["family_best"].isna()
][["model_id", "model_name", "creator", "tags", "downloads"]].copy()
overall_unresolved.to_csv("rq2_unresolved_nodes.csv", index=False)

# -------------------------------------------------------------------
# Best-effort family pair counts
# -------------------------------------------------------------------
print("\n=== BEST-EFFORT FAMILY PAIRS ===")
pair_counter = Counter()

for node_id in multi_ids:
    parents = list(G.predecessors(node_id))
    fams = []
    for p in parents:
        val = nodes.loc[nodes["model_id"] == p, "family_best"]
        if len(val) > 0 and pd.notna(val.iloc[0]):
            fams.append(val.iloc[0])

    unique_fams = sorted(set(fams))
    if len(unique_fams) == 1:
        pair_counter[(unique_fams[0], unique_fams[0])] += 1
    elif len(unique_fams) >= 2:
        for pair in combinations(unique_fams, 2):
            pair_counter[pair] += 1

best_effort_pairs = pd.DataFrame([
    {"family_a": a, "family_b": b, "count": c}
    for (a, b), c in pair_counter.most_common(50)
])

best_effort_pairs.to_csv("rq2_best_effort_family_pairs.csv", index=False)
print(best_effort_pairs.head(15).to_string(index=False))

# -------------------------------------------------------------------
# Quick diagnostic of base_model tags
# -------------------------------------------------------------------
nodes["base_model_tag"] = nodes["tags"].apply(extract_base_model_from_tags)

print("\n=== BASE_MODEL TAG CHECK ===")
print(f"Nodes with base_model tag: {nodes['base_model_tag'].notna().sum():,} / {len(nodes):,}")
print(f"Multi-parent nodes with base_model tag: {multi_subset['base_model_tag'].notna().sum():,} / {len(multi_subset):,}")

# -------------------------------------------------------------------
# Save comparison tables
# -------------------------------------------------------------------
comparison_df = coverage_df.merge(multi_coverage_df, on="method", how="left")
comparison_df.to_csv("rq2_family_method_comparison.csv", index=False)
resolution_df.to_csv("rq2_resolution_by_method.csv", index=False)

summary = f"""
RQ2 DIAGNOSTIC SUMMARY
======================

Graph:
  Nodes: {G.number_of_nodes():,}
  Edges: {G.number_of_edges():,}
  Multi-parent nodes: {len(multi_ids):,}

Coverage by method:
{coverage_df.to_string(index=False)}

Multi-parent coverage by method:
{multi_coverage_df.to_string(index=False)}

Same / cross / unresolved by method:
{resolution_df.to_string(index=False)}

Base model tag signal:
  Nodes with base_model tag: {nodes['base_model_tag'].notna().sum():,} / {len(nodes):,}
  Multi-parent nodes with base_model tag: {multi_subset['base_model_tag'].notna().sum():,} / {len(multi_subset):,}

Interpretation guide:
  - If family_tag performs much better than family_name and family_creator, the issue is mostly our classifier.
  - If family_best still leaves many unresolved nodes, the data itself is genuinely heterogeneous.
  - Inspect rq2_unresolved_multi_parent_examples.csv to see what kinds of models are still hard to classify.
"""

with open("rq2_diagnostic_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("\n" + summary)
print("Saved:")
print("  rq2_family_method_comparison.csv")
print("  rq2_resolution_by_method.csv")
print("  rq2_unresolved_nodes.csv")
print("  rq2_unresolved_multi_parent_examples.csv")
print("  rq2_best_effort_family_pairs.csv")
print("  rq2_diagnostic_summary.txt")