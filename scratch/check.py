import pandas as pd

# Khai báo từ cần tìm
target_word = 'suite'

df = pd.read_parquet('data.parquet')

# Tìm chính xác từ đó (không phân biệt hoa thường)
rows = df[df['word'].str.lower() == target_word.lower()]

with open('scratch/check_result.txt', 'w', encoding='utf-8') as f:
    if rows.empty:
        f.write(f"Không tìm thấy từ '{target_word}' trong DB.\n")
    else:
        for _, row in rows.iterrows():
            d = row.to_dict()
            if 'embedding' in d:
                d['embedding'] = '...'
            f.write(str(d) + '\n\n')
