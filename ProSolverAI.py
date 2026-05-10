
import pandas as pd
import numpy as np
import torch
import os
import re
import json
import threading

from sentence_transformers import SentenceTransformer, util
from colorama import Fore, Style

# Tối ưu hóa môi trường PyTorch cho CPU Ryzen 5 (6 nhân thực)
torch.set_num_threads(6) 
torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tối ưu hóa số lượng ứng viên để đạt tốc độ cực nhanh (< 150ms với MPNet)
MAX_SEMANTIC_CANDIDATES = 100
ENCODE_BATCH = 128

_db_lock = threading.Lock()

class ProSolverAI:

    def __init__(self, db_path="data.parquet", model_name=None):
        self.db_path = db_path
        self._cache = {} # Lưu cache embeddings cho round hiện tại
        
        # Load model name from config if not provided
        if not model_name:
            from config import load_config
            model_name = load_config().get("model", "paraphrase-multilingual-MiniLM-L12-v2")
            
        print(f"  [AI] Khoi dong ProSolverAI ({model_name})...")
        self.model = SentenceTransformer(model_name).to(device)
        
        # Lượng tử hóa model (Chỉ áp dụng trên CPU để tăng tốc 2-3 lần)
        if device == 'cpu':
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("  [AI] Da kich hoat Dynamic Quantization (INT8).")
            except Exception as e:
                print(f"  [!] Khong the luong tu hoa: {e}")

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Khong tim thay DB: {db_path}")

        print("  [DB] Dang nap parquet...")
        self.vocab_db = pd.read_parquet(db_path)

        # Pre-process database
        self.vocab_db['word'] = self.vocab_db['word'].astype(str).str.lower().str.strip()
        # word_only: Loại bỏ khoảng trắng và dấu gạch ngang để khớp với số lượng ô (boxes)
        self.vocab_db['word_only'] = self.vocab_db['word'].str.replace(r'[\s\-]', '', regex=True)
        
        def flatten_meaning(x):
            try:
                data = json.loads(x)
                res = []
                for v in data.values():
                    if isinstance(v, list): res.extend(v)
                return ", ".join(res)
            except: return str(x)

        self.vocab_db['hint_vi'] = self.vocab_db['meaning'].apply(flatten_meaning).fillna("").astype(str)
        
        def get_pos_list(x):
            try: return list(json.loads(x).keys())
            except: return []
        self.vocab_db['pos_list'] = self.vocab_db['meaning'].apply(get_pos_list)
        self.vocab_db['char_length'] = self.vocab_db['word_only'].str.len()
        self.vocab_db['word_structure'] = self.vocab_db['word'].apply(lambda x: [len(part) for part in re.split(r'[\s\-]', str(x)) if part])
        
        self.words_no_space = self.vocab_db['word_only']
        
        # Nạp danh sách Oxford 3000 để ưu tiên từ phổ biến
        self.oxford_words = set()
        oxford_path = os.path.join(os.path.dirname(__file__), "oxford3000.txt")
        if os.path.exists(oxford_path):
            try:
                with open(oxford_path, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip().lower()
                        if w and len(w) > 1: self.oxford_words.add(w)
                print(f"  [AI] Da nap {len(self.oxford_words)} tu Oxford 3000.")
            except: pass

        print(f"  [OK] Turbo DB Ready: {len(self.vocab_db):,} words")

    def solve(self, hint_vi, length, pattern, word_structure=None, example_en="", translation_vi="", pos_tag="", tried_words=[]):
        # DANH SÁCH TỪ DỪNG (STOP WORDS) TIẾNG VIỆT ĐỂ LỌC NHIỄU
        STOP_WORDS = {"điều", "này", "cho", "một", "các", "của", "với", "trong", "khi", "bạn", "làm", "những", "cái", "đang", "được", "việc", "kế", "bên", "hành", "trình", "sự", "như", "nào", "biết", "tại", "là"}
        
        # 1. LOC THEO DO DAI (word_only)
        mask = (self.vocab_db['char_length'] == length)
        if word_structure:
            mask &= self.vocab_db['word_structure'].apply(lambda x: x == word_structure)
        
        # 2. LOC THEO PATTERN (Regex) - Bắt buộc phải khớp
        regex_p = f"^{pattern}$"
        print(f"  {Fore.MAGENTA}[AI]{Style.RESET_ALL} Filter Pattern: {Fore.WHITE}{pattern}")
        # Khớp regex với word_only (không khoảng trắng/gạch ngang)
        candidates = self.vocab_db[mask & self.words_no_space.str.match(regex_p, na=False)].copy()

        # 3. LOẠI BỎ TỪ ĐÃ ĐOÁN (TRIED WORDS) - Chống lặp lại tuyệt đối
        if tried_words and not candidates.empty:
            # Chuẩn hóa tried_words: viết thường, xóa dấu cách/gạch ngang
            tried_normalized = [re.sub(r'[\s\-]', '', str(w).lower()) for w in tried_words]
            # Lọc candidates dựa trên word_only (words_no_space)
            # Chúng ta dùng .loc để đảm bảo khớp đúng index
            mask_tried = self.words_no_space.isin(tried_normalized)
            candidates = candidates[~candidates.index.isin(self.vocab_db[mask_tried].index)]

        if candidates.empty: return pd.DataFrame()

        if len(candidates) > 100: 
            # Lọc sơ bộ 100 từ tiềm năng nhất để đạt tốc độ tối đa
            combined_context = f"{translation_vi} {hint_vi}".lower()
            tokens = set([t for t in re.split(r"\W+", combined_context) if len(t) >= 3 and t not in STOP_WORDS])
            
            if tokens:
                def quick_score(hint):
                    h = str(hint).lower()
                    return sum(1 for t in tokens if t in h)
                
                candidates = candidates.assign(_ov = candidates['hint_vi'].apply(quick_score))
                candidates = candidates.sort_values("_ov", ascending=False).head(100)
            else:
                candidates = candidates.head(100)

        # 3. BATCH ENCODE INPUTS (Tách Nghĩa, Giải thích, Ví dụ EN)
        inputs = []
        has_trans = bool(translation_vi)
        has_hint = bool(hint_vi)
        has_en = bool(example_en)
        
        if has_trans: inputs.append(str(translation_vi))
        if has_hint: inputs.append(str(hint_vi))
        if has_en: inputs.append(example_en.replace("*", " "))
        
        if not inputs: return pd.DataFrame()
        
        all_input_embs = self.model.encode(inputs, convert_to_tensor=True, show_progress_bar=False)
        
        # Mapping embeddings
        curr_idx = 0
        trans_emb = all_input_embs[curr_idx] if has_trans else None
        if has_trans: curr_idx += 1
        
        hint_emb = all_input_embs[curr_idx] if has_hint else None
        if has_hint: curr_idx += 1
        
        en_emb = all_input_embs[curr_idx] if has_en else None

        # 4. CANDIDATE SEMANTICS WITH CACHING
        # Key cache dựa trên tập hợp candidates hiện tại
        cache_key = hash(tuple(candidates['word'].tolist()))
        
        if cache_key in self._cache:
            cand_embs, word_embs = self._cache[cache_key]
        else:
            # Encode meanings
            cand_embs = self.model.encode(
                candidates['hint_vi'].tolist(), 
                convert_to_tensor=True, 
                show_progress_bar=False, 
                batch_size=ENCODE_BATCH
            )
            # Encode English words
            if has_en:
                cand_words_list = candidates['word'].tolist()
                word_embs = self.model.encode(
                    cand_words_list, 
                    convert_to_tensor=True, 
                    show_progress_bar=False, 
                    batch_size=ENCODE_BATCH
                )
            else:
                word_embs = torch.zeros((len(candidates), cand_embs.shape[1]), device=device)
            
            # Lưu vào cache (Giới hạn size cache để tránh tốn RAM)
            if len(self._cache) > 10: self._cache.clear()
            self._cache[cache_key] = (cand_embs, word_embs)

        # Tính điểm cho từng thành phần
        s_trans_vals = util.cos_sim(trans_emb, cand_embs)[0] if has_trans else torch.zeros(len(candidates), device=device)
        s_hint_vals = util.cos_sim(hint_emb, cand_embs)[0] if has_hint else torch.zeros(len(candidates), device=device)
        s_en_vals = util.cos_sim(en_emb, word_embs)[0] if has_en else torch.zeros(len(candidates), device=device)

        candidates = candidates.assign(
            s_trans = s_trans_vals.detach().cpu().numpy(),
            s_hint = s_hint_vals.detach().cpu().numpy(),
            s_en = s_en_vals.detach().cpu().numpy()
        )
        
        # 4. TINH TOAN DIEM SO TONG HOP (Cấu trúc mới: Semantic là NỀN TẢNG)
        h_low = str(hint_vi).lower()
        t_low = str(translation_vi).lower().strip()
        is_hidden_meaning = "nghĩa ẩn" in t_low or not t_low
        
        if is_hidden_meaning:
            # Nếu nghĩa ẩn: Tin 100% vào Giải thích (s_hint)
            candidates = candidates.assign(
                final_score = (candidates['s_hint'] * 5.0).astype(np.float32)
            )
        else:
            # s_trans chiếm 70% trọng số ngữ nghĩa, s_hint 30%
            candidates = candidates.assign(
                final_score = ((0.70 * candidates['s_trans'] + 
                                0.30 * candidates['s_hint']) * 5.0).astype(np.float32)
            )

        # 5. VECTORIZED BONUSES
        # ...

        # 5. VECTORIZED BONUSES (Tinh chinh lai trong so)
        
        # Chuan hoa POS tag de khop voi DB (VD: ADJECTIVE -> adj, a, adjective)
        pos_map = {
            "adjective": ["adj", "a", "adjective"],
            "noun": ["n", "noun", "s"],
            "verb": ["v", "verb"],
            "adverb": ["adv", "adverb", "r"],
            "pronoun": ["pron", "pronoun"],
            "preposition": ["prep", "preposition"],
            "conjunction": ["conj", "conjunction"],
            "determiner": ["det", "determiner"],
            "idiom": ["idiom", "idm"],
            "phrase": ["phrase", "phr"]
        }
        target_pos_list = pos_map.get(pos_tag.lower(), [pos_tag.lower()])

        # THƯỞNG 1: KEYWORD INTERSECTION (PHÂN CẤP VÀNG - BẠC)
        # Từ khóa VÀNG: Giữ nguyên cụm từ (VD: "thư từ", "canh tác") - Chỉ tách theo dấu chấm phẩy/phẩy
        t_keywords = list(set([w.strip() for w in re.split(r'[;,]+', t_low) if len(w.strip()) >= 3 and w.strip() not in STOP_WORDS]))
        # Từ khóa BẠC: Từ phần giải thích (Vẫn tách lẻ để tìm gợi ý)
        h_keywords = list(set([w for w in re.split(r'[\s,.\-?]+', h_low) if len(w) >= 3 and w not in STOP_WORDS]))
        
        def count_matches(db_hint, kws):
            db_h = f" {str(db_hint).lower()} "
            # Kiểm tra khớp nguyên cụm để tránh "thư từ" bị tách thành "từ"
            return sum(1 for kw in kws if f" {kw} " in db_h)

        if t_keywords or h_keywords:
            t_match_counts = candidates['hint_vi'].apply(lambda x: count_matches(x, t_keywords)).astype(np.float32)
            h_match_counts = candidates['hint_vi'].apply(lambda x: count_matches(x, h_keywords)).astype(np.float32)
            
            # Thưởng từ khóa: Nếu nghĩa ẩn, đẩy mạnh thưởng từ giải thích (+1.0)
            h_boost = 1.0 if is_hidden_meaning else 0.2
            candidates.loc[:, 'final_score'] += t_match_counts * 0.8
            candidates.loc[:, 'final_score'] += h_match_counts * h_boost

        # THƯỞNG 2: ENGLISH RIDDLE CONTEXT (MẠNH HƠN)
        eng_in_riddle = [w for w in re.findall(r'[a-zA-Z]{4,}', h_low)]
        if eng_in_riddle:
            def count_eng_matches(db_hint):
                db_h = f" {str(db_hint).lower()} "
                return sum(1 for ew in eng_in_riddle if f" {ew} " in db_h)
            
            eng_match_counts = candidates['hint_vi'].apply(count_eng_matches).astype(np.float32)
            candidates.loc[:, 'final_score'] += eng_match_counts * 0.6

        # THƯỞNG 3: LOẠI TỪ (POS TAG) - QUYẾT ĐỊNH MẠNH
        if pos_tag:
            is_pos_match = candidates['pos_list'].apply(lambda x: any(p in x for p in target_pos_list))
            has_pos_info = candidates['pos_list'].apply(lambda x: len(x) > 0)
            
            # Phạt nặng từ sai loại từ (ví dụ: đề yêu cầu Động từ mà lại là Danh từ)
            candidates.loc[has_pos_info & ~is_pos_match, 'final_score'] -= 1.5
            # Thưởng mạnh cho từ đúng loại từ
            candidates.loc[is_pos_match, 'final_score'] += 1.0
            # Phạt nhẹ từ không có thông tin loại từ để ưu tiên từ có thông tin khớp
            candidates.loc[~has_pos_info, 'final_score'] -= 0.5
        
        # THƯỞNG 4: VÍ DỤ TIẾNG ANH (HÌNH ẢNH HÓA)
        if has_en:
            en_keywords = [w for w in example_en.replace("*", " ").lower().split() if len(w) > 4]
            if en_keywords:
                kw_p = "|".join([re.escape(k) for k in en_keywords])
                candidates.loc[candidates['hint_vi'].str.contains(kw_p, na=False), 'final_score'] += 0.5

        # 6. POPULARITY BONUSES (TĂNG MẠNH CHO OXFORD)
        if self.oxford_words:
            # Các từ thông dụng thường là đáp án đúng trong các game vocab
            candidates.loc[candidates['word'].isin(self.oxford_words), 'final_score'] += 0.5
        
        # 7. COMPOUND WORD BOOST: Ưu tiên Khoảng trắng trước, Dấu gạch ngang sau
        if len(word_structure) > 1:
            split_idx = word_structure[0]
            # Thưởng mạnh cho Khoảng trắng (Cụm từ thông dụng)
            is_space_match = candidates['word'].apply(lambda x: len(x) > split_idx and x[split_idx] == ' ')
            candidates.loc[is_space_match, 'final_score'] += 0.8
            
            # Thưởng vừa cho Dấu gạch ngang (Từ ghép)
            is_hyphen_match = candidates['word'].apply(lambda x: len(x) > split_idx and x[split_idx] == '-')
            candidates.loc[is_hyphen_match, 'final_score'] += 0.4
        
        if 'id' in candidates.columns:
            candidates = candidates.assign(
                id_val = pd.to_numeric(candidates['id'], errors='coerce').fillna(999999)
            )
            candidates = candidates.assign(
                final_score = (candidates['final_score'] + (10000 / (candidates['id_val'] + 10000)) * 0.05).astype(np.float32)
            )

        # 7. TRUY QUÉT TỪ TIẾNG ANH TRONG CONTEXT TIẾNG VIỆT (FUZZY MATCHING)
        # Gộp các nguồn văn bản, lọc bỏ giá trị rỗng/null
        sources = [str(hint_vi or ""), str(translation_vi or "")]
        combined_vi_context = " ".join(sources).lower()
        
        words_in_vi = set(re.findall(r'\b[a-z]{3,}\b', combined_vi_context)) 
        
        if words_in_vi and not candidates.empty:
            from difflib import SequenceMatcher
            for context_word in words_in_vi:
                # Chỉ xét các từ trong context có độ dài gần khớp (chênh lệch tối đa 2 ký tự)
                if abs(len(context_word) - length) <= 2:
                    for idx, row in candidates.iterrows():
                        cand_word = str(row['word'])
                        # Tính độ tương đồng
                        sim = SequenceMatcher(None, context_word, cand_word).ratio()
                        if sim >= 0.85: # Ngưỡng tương đồng cao
                            if sim < 1.0:
                                print(f"  {Fore.MAGENTA}[AI]{Style.RESET_ALL} Phat hien tu tuong tu trong context: {Fore.YELLOW}{context_word}{Style.RESET_ALL} -> {Fore.GREEN}{cand_word}{Style.RESET_ALL} (Sim: {sim:.2f})")
                            else:
                                print(f"  {Fore.MAGENTA}[AI]{Style.RESET_ALL} Khop tu trong context: {Fore.GREEN}{cand_word}")
                            
                            candidates.at[idx, 'final_score'] += 2.5

        # 8. LOGGING TOP 3 FOR DEBUGGING
        if not candidates.empty:
            top_3 = candidates.sort_values('final_score', ascending=False).head(3)
            print(f"  {Fore.CYAN}--- AI Analysis (Top 3) ---")
            for _, r in top_3.iterrows():
                print(f"  Word: {Fore.WHITE}{r['word']:<12}{Style.RESET_ALL} | Score: {Fore.GREEN}{r['final_score']:.4f}{Style.RESET_ALL} (T:{r.get('s_trans',0):.2f} H:{r.get('s_hint',0):.2f} E:{r.get('s_en',0):.2f})")

        return candidates.sort_values('final_score', ascending=False).head(5)

    def learn_new_case(self, word, hint_vi, word_structure=None, pos_tag=""):
        word, hint_vi = str(word).lower().strip(), str(hint_vi).strip()
        pos_tag = str(pos_tag).lower().strip() or "learned"
        
        if not word or not hint_vi: return

        with _db_lock:
            existing_mask = (self.vocab_db['word'] == word)
            if existing_mask.any():
                idx = self.vocab_db[existing_mask].index[0]
                
                # 1. Cập nhật JSON meaning
                try:
                    m_raw = self.vocab_db.at[idx, 'meaning']
                    meaning_json = json.loads(m_raw) if m_raw else {}
                except:
                    meaning_json = {}
                
                if pos_tag not in meaning_json:
                    meaning_json[pos_tag] = []
                
                if hint_vi not in meaning_json[pos_tag]:
                    meaning_json[pos_tag].append(hint_vi)
                    
                    # 2. Lưu lại JSON và cập nhật các trường phái sinh
                    self.vocab_db.at[idx, 'meaning'] = json.dumps(meaning_json, ensure_ascii=False)
                    
                    all_hints = []
                    for v in meaning_json.values():
                        if isinstance(v, list): all_hints.extend(v)
                    self.vocab_db.at[idx, 'hint_vi'] = ", ".join(list(set(all_hints)))
                    self.vocab_db.at[idx, 'pos_list'] = list(meaning_json.keys())
                    
                    # Lưu xuống file
                    self.vocab_db.to_parquet(self.db_path, index=False)
            else:
                # 3. Tạo mới hoàn toàn
                # TỰ TÍNH CẤU TRÚC TỪ CHÍNH TỪ ĐÓ (Chuẩn hơn lấy từ UI bị wrap)
                calculated_ws = [len(part) for part in re.split(r'[\s\-]', word) if part]
                new_row = pd.DataFrame([{
                    "word": word, "ipa": "", "parts": "[]", 
                    "meaning": json.dumps({pos_tag: [hint_vi]}, ensure_ascii=False), 
                    "hint_vi": hint_vi, "char_length": len(word.replace(" ", "").replace("-", "")), 
                    "word_structure": calculated_ws,
                    "pos_list": [pos_tag]
                }])
                self.vocab_db = pd.concat([self.vocab_db, new_row], ignore_index=True)
                # Cập nhật lại words_no_space để không bị lỗi regex
                self.words_no_space = self.vocab_db['word'].str.replace(" ", "", regex=False)
                self.vocab_db.to_parquet(self.db_path, index=False)

    def get_db_size(self): return len(self.vocab_db)

_db_file = "data.parquet"
if not os.path.exists(_db_file): _db_file = os.path.join(os.path.dirname(__file__), "data.parquet")
ai_engine = ProSolverAI(db_path=_db_file)
