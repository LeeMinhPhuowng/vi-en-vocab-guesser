import pandas as pd
import os

db_path = "d:/Vocab Tool/data.parquet"
if os.path.exists(db_path):
    df = pd.read_parquet(db_path)
    res = df[df['word'].str.contains("BEGINNING", case=False)]
    print(res[['word', 'word_structure']])
