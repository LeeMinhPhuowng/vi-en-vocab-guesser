import pandas as pd
import os

db_path = "d:/Vocab Tool/data.parquet"

if os.path.exists(db_path):
    print(f"Loading database from {db_path}...")
    df = pd.read_parquet(db_path)
    
    # Tìm xem có bao nhiêu dòng chứa từ này
    wrong_word = "noncontributory"
    initial_count = len(df)
    
    # Xóa dòng có từ này
    df_cleaned = df[df['word'].str.lower() != wrong_word]
    
    deleted_count = initial_count - len(df_cleaned)
    
    if deleted_count > 0:
        df_cleaned.to_parquet(db_path, index=False)
        print(f"Successfully deleted '{wrong_word}' from database. ({deleted_count} row(s) removed)")
    else:
        print(f"Word '{wrong_word}' not found in database.")
else:
    print(f"Database file not found at {db_path}")
