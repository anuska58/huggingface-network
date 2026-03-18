import pandas as pd
import re
import csv
from tqdm import tqdm

INPUT_FILE = "merged_dataset.csv"
OUTPUT_FILE = "edges.csv"


# ----------------------------
# Extract multiple parents
# ----------------------------
def extract_parents(text):

    if not isinstance(text, str):
        return []

    patterns = [
        r"fine[- ]?tuned from ([\w\-\/\.]+)",
        r"based on ([\w\-\/\.]+)",
        r"derived from ([\w\-\/\.]+)",
        r"initialized from ([\w\-\/\.]+)"
    ]

    parents = []

    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        parents.extend(matches)

    return list(set(parents))


# ----------------------------
# Detect transformation
# ----------------------------
def detect_transformation(tags, model_name, card_text):

    tags = str(tags).lower()

    if any(x in tags for x in ["gptq","gguf","4bit","8bit","quantized"]):
        return "quantization", 0.9

    if "lora" in tags:
        return "adapter", 0.9

    if "merge" in model_name.lower():
        return "merge", 0.8

    if re.search(r"fine[- ]?tuned", str(card_text), re.IGNORECASE):
        return "fine-tune", 0.9

    return "unknown", 0.5


# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(INPUT_FILE)

edges = []

print("Building edges with model cards...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    model_id = row["model_id"]
    model_name = row["model_name"]
    tags = row["tags"]
    card = row.get("model_card", "")

    parents = []

    # -------------------------
    # 1. Extract from model_card
    # -------------------------
    parents = extract_parents(card)

    # -------------------------
    # 2. If no parent → infer
    # -------------------------
    if not parents:

        if any(x in str(tags).lower() for x in ["gptq","gguf","lora"]):

            base_guess = re.sub(r'-(gptq|gguf|lora).*', '', model_name, flags=re.IGNORECASE)

            parents = [f"{row['creator']}/{base_guess}"]

    # -------------------------
    # Create edges
    # -------------------------
    for parent in parents:

        if parent and parent != model_id:

            transformation, confidence = detect_transformation(tags, model_name, card)

            edges.append([
                parent,
                model_id,
                "Directed",
                transformation,
                confidence
            ])


# ----------------------------
# Save edges
# ----------------------------
with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Source",
        "Target",
        "Type",
        "Transformation",
        "Confidence"
    ])

    writer.writerows(edges)


print("Done!")
print("Total edges:", len(edges))