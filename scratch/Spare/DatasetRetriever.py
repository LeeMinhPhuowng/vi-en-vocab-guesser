from datasets import load_dataset, concatenate_datasets
import pandas as pd

ds = load_dataset("KietReal/vietnamese-english-translation")
print(ds)

# print(f"Tổng số câu sau khi gộp: {len(df)}")