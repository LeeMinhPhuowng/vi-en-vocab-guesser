import pandas as pd
import os
import re

db_path = 'data.parquet'
df = pd.read_parquet(db_path)

# Các từ bị học sai (nhặt nhầm từ xung quanh)
wrong_patterns = [
    'CATER FOOD', 'JOURNALIST EVENTS', 'FACILITATE CITIES', 
    'SOLE YOU', 'ANALYSIS RESULTS', 'TRIM GARDEN', 
    'SACRIFICE STUDY', 'TINY TREE', 'LOGO YEAR'
]

initial_count = len(df)
print(f"Initial DB size: {initial_count}")

# 1. Xóa các từ nằm trong danh sách sai
df = df[~df['word'].str.upper().isin(wrong_patterns)]

# 2. Logic dọn dẹp chung: Nếu từ có khoảng trắng nhưng hint_vi chỉ là 1 từ đơn 
# và không phải là cụm từ cố định nổi tiếng, thường là do học nhầm.
# Tuy nhiên để an toàn, tôi sẽ chỉ xử lý các từ vừa học gần đây (không có IPA)
mask_new = df['ipa'] == ""
df_new = df[mask_new]

def is_likely_garbage(row):
    word = str(row['word'])
    if " " in word:
        # Nếu từ có khoảng trắng nhưng word_structure chỉ có 1 phần tử (lỗi logic trước đó)
        if len(row['word_structure']) == 1:
            return True
    return False

df = df[~df.apply(is_likely_garbage, axis=1)]

# 3. Đảm bảo các từ đúng được cập nhật nghĩa
corrections = {
    "CATER": "phục vụ",
    "JOURNALIST": "nhà báo",
    "FACILITATE": "tạo điều kiện",
    "SOLE": "duy nhất",
    "ANALYSIS": "phân tích",
    "TRIM": "cắt tỉa",
    "SACRIFICE": "hy sinh",
    "TINY": "tí hon",
    "LOGO": "logo"
}

for word, hint in corrections.items():
    mask = df['word'].str.upper() == word
    if mask.any():
        idx = df[mask].index[0]
        df.at[idx, 'hint_vi'] = hint

print(f"Cleaned DB size: {len(df)} (Removed {initial_count - len(df)} rows)")
df.to_parquet(db_path, index=False)
