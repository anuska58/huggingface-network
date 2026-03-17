import pandas as pd

# NEW dataset (341k models)
df_new = pd.read_csv("all_text_generation_models.csv")

# OLD dataset (contains model_card)
df_old = pd.read_csv("huggingface_models.csv")

# create model_id in old dataset
df_old["model_id"] = df_old["creator"] + "/" + df_old["model_name"]

# keep only needed columns
df_old = df_old[["model_id", "model_card"]]

# merge
df_merged = df_new.merge(df_old, on="model_id", how="left")

# save
df_merged.to_csv("merged_dataset.csv", index=False)

print("Merged dataset created!")
print("Rows:", len(df_merged))
print("Model cards available:", df_merged["model_card"].notna().sum())