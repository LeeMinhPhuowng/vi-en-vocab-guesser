import pandas as pd

# 1. Đọc file CSV đã lọc từ bước trước
df = pd.read_csv("vocab_for_ai.csv")

# 2. Loại bỏ các dòng bị lỗi (nếu có)
df = df.dropna()

# 3. Chuyển cột length về kiểu số nguyên để tìm kiếm nhanh hơn
df['length'] = df['length'].astype(int)

# 4. Lưu sang định dạng Parquet với chuẩn nén 'snappy'
df.to_parquet("vocab_database.parquet", engine="pyarrow", compression="snappy")

print("Đã tạo xong cơ sở dữ liệu Parquet!")