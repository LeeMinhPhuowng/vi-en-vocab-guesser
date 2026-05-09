import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# 1. Load database Parquet
df = pd.read_parquet("vocab_database.parquet")

# 2. Load Model AI hiểu đa ngôn ngữ (Vi-En)
# Model này rất nhẹ, có thể chạy tốt trên CPU
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def find_my_word(user_input_meaning, user_input_length):
    # Bước A: Lọc cứng theo số chữ cái (Cực nhanh với Parquet)
    filtered_df = df[df['length'] == user_input_length].copy()
    
    if filtered_df.empty:
        return "Không tìm thấy từ nào có độ dài này."

    # Bước B: So sánh ngữ nghĩa bằng AI
    # Lấy tất cả các câu gợi ý tiếng Việt của những từ có độ dài phù hợp
    hints = filtered_df['hint_vi'].tolist()
    
    # Biến câu nhập và các gợi ý thành Vector
    user_vec = model.encode(user_input_meaning, convert_to_tensor=True)
    hint_vecs = model.encode(hints, convert_to_tensor=True)
    
    # Tính toán độ tương đồng (Cosine Similarity)
    cosine_scores = util.cos_sim(user_vec, hint_vecs)[0]
    
    # Lấy ra kết quả cao nhất
    top_result_index = torch.argmax(cosine_scores).item()
    
    word_found = filtered_df.iloc[top_result_index]['word']
    confidence = cosine_scores[top_result_index].item()
    
    return {
        "word": word_found,
        "score": round(confidence * 100, 2),
        "example_vi": filtered_df.iloc[top_result_index]['hint_vi']
    }

if __name__ == "__main__":
    # --- TEST THỬ ---
    meaning = "một loại quả màu đỏ, thường ăn rất giòn"
    length = 5
    result = find_my_word(meaning, length)

    print(f"Từ AI tìm được: {result['word']} (Độ chính xác: {result['score']}%)")
    print(f"Ngữ cảnh khớp: {result['example_vi']}")