import pandas as pd
import spacy
from datasets import load_dataset
from tqdm import tqdm
import gc
import re

# Tải model core_web_sm, disable các thành phần không cần thiết để tăng tốc
nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])

def get_char_length(word):
    # Loại bỏ space nhưng GIỮ gạch ngang (vì game tính gạch ngang là 1 ô)
    return len(word.replace(" ", ""))

def is_clean_english(text):
    # Chấp nhận Latin, số (cho ngữ cảnh), dấu cách, gạch ngang và các dấu câu cơ bản
    return bool(re.match(r"^[a-zA-Z0-9\s\-\.,!\?']+$", text))

def process_data(max_samples=None):
    print("Khởi động bộ lọc v2: Sạch hơn - Thông minh hơn...")
    dataset = load_dataset("KietReal/vietnamese-english-translation", split='train', streaming=True)
    
    word_counts = {}
    word_to_hint = {}
    BATCH_SIZE = 2000
    dataset_iter = iter(dataset)
    count = 0

    total_rows = dataset.info.splits['train'].num_examples

    actual_limit = min(max_samples, total_rows) if max_samples else total_rows

    print(f"Tổng cộng có {total_rows:,} câu. Bắt đầu xử lý {actual_limit:,} câu...")
    # Pattern: Chấp nhận chữ cái, có thể có gạch ngang hoặc dấu cách ở giữa
    # Ví dụ: "apple", "self-aware", "ice cream"
    valid_pattern = re.compile(r"^[a-z]+(?:[ \-][a-z]+)*$")

    try:
        with tqdm(total=actual_limit if max_samples else 2700000, desc="Processing") as pbar:
            while True:
                current_en_batch, current_vi_batch = [], []
                try:
                    for _ in range(BATCH_SIZE):
                        ex = next(dataset_iter)
                        current_en_batch.append(str(ex['English']))
                        current_vi_batch.append(str(ex['Vietnamese']))
                except StopIteration: break

                # NLP Pipe xử lý đa luồng ngầm định
                for i, doc in enumerate(nlp.pipe(current_en_batch, batch_size=BATCH_SIZE)):
                    vi_context = current_vi_batch[i].strip()
                    en_sentence = current_en_batch[i]

                    # 1. Lọc nhanh đầu nguồn
                    if len(vi_context) > 120 or not is_clean_english(en_sentence):
                        continue

                    # 2. Nhận diện tên riêng để loại bỏ
                    # Thêm 'CARDINAL', 'DATE' để tránh lấy số lượng/ngày tháng làm từ vựng
                    proper_names = {ent.text.lower() for ent in doc.ents 
                                   if ent.label_ in ["PERSON", "GPE", "ORG", "LOC", "FAC", "PRODUCT"]}
                    
                    extracted_candidate = set()

                    # A. Trích xuất Cụm danh từ (Noun Chunks) - Cho từ ghép
                    for chunk in doc.noun_chunks:
                        text = chunk.text.lower().strip()
                        # Loại bỏ mạo từ
                        text = re.sub(r"^(a|an|the)\s+", "", text).strip()
                        
                        if (1 < len(text.split()) <= 3 and 
                            text not in proper_names and 
                            valid_pattern.match(text)):
                            extracted_candidate.add(text)

                    # B. Trích xuất Token (Từ đơn & Phrasal Verbs)
                    for token in doc:
                        word_low = token.text.lower()
                        
                        # Loại rác: Danh từ riêng, Stopwords, Tên riêng NER, Ký tự lạ
                        if (token.pos_ == "PROPN" or token.is_stop or 
                            word_low in proper_names or len(word_low) < 3):
                            continue

                        # Xử lý Phrasal Verbs
                        if token.pos_ == "VERB":
                            for child in token.children:
                                if child.dep_ in ["prt", "prep"]:
                                    phrasal = f"{word_low} {child.text.lower()}"
                                    if valid_pattern.match(phrasal):
                                        extracted_candidate.add(phrasal)

                        # Xử lý từ đơn (chỉ lấy Noun, Verb, Adj, Adv)
                        if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                            if valid_pattern.match(word_low):
                                extracted_candidate.add(word_low)

                    # 3. Cập nhật Dictionary (Chiến thuật: Câu giải thích ngắn nhất là tốt nhất)
                    for w in extracted_candidate:
                        word_counts[w] = word_counts.get(w, 0) + 1
                        if w not in word_to_hint or len(vi_context) < len(word_to_hint[w]):
                            word_to_hint[w] = vi_context

                count += len(current_en_batch)
                pbar.update(len(current_en_batch))
                if count % 100000 == 0: gc.collect()
                if max_samples and count >= max_samples: break

    except Exception as e:
        print(f"\nLỗi trong quá trình xử lý: {e}")

    print("\nĐang tổng hợp kết quả...")
    
    # Lọc lần cuối: Chỉ lấy từ xuất hiện >= 5 lần để đảm bảo không phải typo
    final_data = [
        {
            "word": w, 
            "length": len(w), # Độ dài gồm cả space
            "char_length": get_char_length(w), # Số ô thực tế trong game
            "hint_vi": word_to_hint[w], 
            "frequency": c
        }
        for w, c in word_counts.items() if c >= 5
    ]
    
    if not final_data:
        print("Không có dữ liệu sau khi lọc!")
        return

    df = pd.DataFrame(final_data)
    
    # Tối ưu kiểu dữ liệu để giảm dung lượng file Parquet
    df['length'] = df['length'].astype('int16')
    df['char_length'] = df['char_length'].astype('int16')
    df['frequency'] = df['frequency'].astype('int32')

    df = df.sort_values(by="frequency", ascending=False)
    
    output_file = "vocab_intelligent.parquet"
    df.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"Hoàn thành! Đã lưu {len(df)} từ chất lượng vào {output_file}")

if __name__ == "__main__":
    # Khuyến khích chạy thử 100k câu trước khi chạy full
    # process_data(max_samples=100000) 
    process_data(max_samples=None)