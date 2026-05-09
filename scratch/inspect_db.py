import pandas as pd
import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

df = pd.read_parquet('data.parquet')
run_row = df[df['word'] == 'run']
if not run_row.empty:
    print(run_row.iloc[0]['meaning'])
else:
    print("Word 'run' not found")
