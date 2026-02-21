import pandas as pd

df=pd.read_csv("model_metadata.csv")
print(df["url"].duplicated().sum())