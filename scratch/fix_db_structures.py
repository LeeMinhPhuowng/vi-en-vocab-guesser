import pandas as pd
import os
import re

db_path = "d:/Vocab Tool/data.parquet"

if os.path.exists(db_path):
    print(f"Checking database integrity...")
    df = pd.read_parquet(db_path)
    
    modified = False
    
    def fix_ws(row):
        global modified
        word = str(row['word'])
        # Tự tính cấu trúc chuẩn từ từ vựng
        correct_ws = [len(part) for part in re.split(r'[\s\-]', word) if part]
        
        # Nếu cấu trúc hiện tại khác cấu trúc chuẩn
        if list(row['word_structure']) != correct_ws:
            print(f"  Fixing structure for '{word}': {row['word_structure']} -> {correct_ws}")
            modified = True
            return correct_ws
        return row['word_structure']

    df['word_structure'] = df.apply(fix_ws, axis=1)
    
    if modified:
        df.to_parquet(db_path, index=False)
        print("Database structure updated successfully.")
    else:
        print("All structures are already correct.")
else:
    print(f"Database file not found at {db_path}")
