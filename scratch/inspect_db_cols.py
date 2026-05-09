import pandas as pd
df = pd.read_parquet("d:/Vocab Tool/data.parquet")
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
