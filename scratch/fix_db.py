import pandas as pd
import json
import os

db_path = 'data.parquet'
df = pd.read_parquet(db_path)

# 1. Tìm và sửa 'segue'
segue_mask = df['word'].str.lower() == 'segue'
if segue_mask.any():
    print("Found 'segue'. Removing incorrect meaning...")
    idx = df[segue_mask].index[0]
    # Trả về nghĩa chuẩn cho segue (sự chuyển tiếp)
    df.at[idx, 'meaning'] = json.dumps({"noun": ["sự chuyển tiếp", "sự liên tục"]}, ensure_ascii=False)
    df.at[idx, 'hint_vi'] = "sự chuyển tiếp, sự liên tục"

# 2. Đảm bảo 'stage' có nghĩa đúng
stage_mask = df['word'].str.lower() == 'stage'
meaning_stage = json.dumps({
    "noun": ["giai đoạn", "bước", "sân khấu"],
    "verb": ["tổ chức", "dàn dựng"]
}, ensure_ascii=False)

if stage_mask.any():
    print("Updating 'stage'...")
    idx = df[stage_mask].index[0]
    df.at[idx, 'meaning'] = meaning_stage
    df.at[idx, 'hint_vi'] = "giai đoạn, bước, sân khấu"
else:
    print("Adding 'stage'...")
    new_row = pd.DataFrame([{
        "word": "stage",
        "ipa": "/steɪdʒ/",
        "parts": "[]",
        "meaning": meaning_stage,
        "hint_vi": "giai đoạn, bước, sân khấu",
        "char_length": 5,
        "word_structure": [5],
        "pos_list": ["noun", "verb"]
    }])
    df = pd.concat([df, new_row], ignore_index=True)

# Lưu lại
df.to_parquet(db_path, index=False)
print("DB fixed successfully!")
