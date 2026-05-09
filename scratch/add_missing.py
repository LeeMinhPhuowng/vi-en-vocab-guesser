import pandas as pd
import json
import os

db_path = 'data.parquet'
df = pd.read_parquet(db_path)

new_entries = [
    {
        "word": "dog tired",
        "ipa": "/ˌdɒɡ ˈtaɪəd/",
        "parts": "[]",
        "meaning": json.dumps({"adj": ["mệt nhoài", "rất mệt"]}, ensure_ascii=False),
        "hint_vi": "mệt nhoài",
        "char_length": 8,
        "word_structure": [3, 5],
        "pos_list": ["adj"]
    },
    {
        "word": "have the green fingers",
        "ipa": "",
        "parts": "[]",
        "meaning": json.dumps({"idiom": ["mát tay (trồng trọt)"]}, ensure_ascii=False),
        "hint_vi": "mát tay",
        "char_length": 19,
        "word_structure": [4, 3, 5, 7],
        "pos_list": ["idiom"]
    },
    {
        "word": "have got green fingers",
        "ipa": "",
        "parts": "[]",
        "meaning": json.dumps({"idiom": ["mát tay (trồng trọt)"]}, ensure_ascii=False),
        "hint_vi": "mát tay",
        "char_length": 19,
        "word_structure": [4, 3, 5, 7],
        "pos_list": ["idiom"]
    }
]

for entry in new_entries:
    # Xóa bản cũ nếu có
    df = df[df['word'].str.lower() != entry['word'].lower()]
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

df.to_parquet(db_path, index=False)
print("Manual entries added successfully!")
