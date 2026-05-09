
import pandas as pd
import numpy as np
import torch
import os
import re
import json
import threading

from sentence_transformers import SentenceTransformer, util

# Tối ưu hóa môi trường PyTorch cho tốc độ suy luận
torch.set_num_threads(4) 
torch.set_grad_enabled(False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tối ưu hóa số lượng ứng viên để đạt tốc độ cực nhanh (< 200ms)
MAX_SEMANTIC_CANDIDATES = 150
ENCODE_BATCH = 256

_db_lock = threading.Lock()

class ProSolverAI:

    def __init__(self, db_path="data.parquet"):
        self.db_path = db_path
        print(f"  [AI] Khoi dong ProSolverAI TURBO EDITION...")
        
        # Nạp model
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(device)
        
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
        self.vocab_db['word_only'] = self.vocab_db['word'].str.replace(" ", "", regex=False)
        
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
        self.vocab_db['word_structure'] = self.vocab_db['word'].apply(lambda x: [len(part) for part in str(x).split() if part])
        
        self.words_no_space = self.vocab_db['word_only']
        print(f"  [OK] Turbo DB Ready: {len(self.vocab_db):,} words")

    def solve(self, hint_vi, length, pattern, word_structure=None, example_en="", example_vi="", pos_tag="", tried_words=[]):
        # 1. HARD FILTER (FAST)
        mask = (self.vocab_db['char_length'] == length)
        if word_structure:
            mask &= self.vocab_db['word_structure'].apply(lambda x: x == word_structure)
        
        regex_p = f"^{pattern}$"
        candidates = self.vocab_db[mask & self.words_no_space.str.match(regex_p, na=False)].copy()

        if candidates.empty:
            if pattern != "." * length:
                candidates = self.vocab_db[mask].copy()
            else:
                return pd.DataFrame()

        # Loại bỏ các từ đã thử ngay từ đầu để giảm tải cho AI
        if tried_words:
            candidates = candidates[~candidates['word'].isin(tried_words)]

        if candidates.empty: return pd.DataFrame()

        # 2. VECTORIZED TOKEN OVERLAP (FAST FILTER)
        if len(candidates) > MAX_SEMANTIC_CANDIDATES:
            tokens = [t for t in re.split(r"\W+", str(hint_vi).lower()) if len(t) >= 2]
            if tokens:
                pattern_tokens = "|".join([re.escape(t) for t in tokens])
                candidates['_ov'] = candidates['hint_vi'].str.count(pattern_tokens)
                candidates = candidates.sort_values("_ov", ascending=False).head(MAX_SEMANTIC_CANDIDATES)
            else:
                candidates = candidates.head(MAX_SEMANTIC_CANDIDATES)

        # 3. BATCH ENCODE INPUTS
        inputs = [str(hint_vi)]
        has_vi, has_en = bool(example_vi), bool(example_en)
        if has_vi: inputs.append(str(example_vi))
        if has_en: inputs.append(example_en.replace("*", " "))
        
        all_input_embs = self.model.encode(inputs, convert_to_tensor=True, show_progress_bar=False)
        hint_emb, vi_emb, en_emb = all_input_embs[0], None, None
        if has_vi: vi_emb = all_input_embs[1]
        if has_en: en_emb = all_input_embs[-1]

        # 4. CANDIDATE SEMANTICS
        cand_embs = self.model.encode(
            candidates['hint_vi'].tolist(), 
            convert_to_tensor=True, 
            show_progress_bar=False, 
            batch_size=ENCODE_BATCH
        )

        # Matrix Multiplication
        s_hint = util.cos_sim(hint_emb, cand_embs)[0]
        s_vi = util.cos_sim(vi_emb, cand_embs)[0] if has_vi else torch.zeros(len(candidates), device=device)
        s_en = util.cos_sim(en_emb, cand_embs)[0] if has_en else torch.zeros(len(candidates), device=device)

        candidates['final_score'] = (0.30 * s_hint + 0.30 * s_vi + 0.40 * s_en).detach().cpu().numpy()

        # 5. VECTORIZED BONUSES
        h_low = str(hint_vi).lower()
        candidates.loc[candidates['hint_vi'].str.contains(re.escape(h_low), na=False), 'final_score'] += 1.0
        if pos_tag:
            candidates.loc[candidates['pos_list'].apply(lambda x: pos_tag in x), 'final_score'] += 0.5
        if has_en:
            en_keywords = [w for w in example_en.replace("*", " ").lower().split() if len(w) > 4]
            if en_keywords:
                kw_p = "|".join([re.escape(k) for k in en_keywords])
                candidates.loc[candidates['hint_vi'].str.contains(kw_p, na=False), 'final_score'] += 0.6

        return candidates.sort_values('final_score', ascending=False).head(5)

    def learn_new_case(self, word, hint_vi, example_vi="", word_structure=None, pos_tag=""):
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
                new_row = pd.DataFrame([{
                    "word": word, "ipa": "", "parts": "[]", 
                    "meaning": json.dumps({pos_tag: [hint_vi]}, ensure_ascii=False), 
                    "hint_vi": hint_vi, "char_length": len(word.replace(" ", "")), 
                    "word_structure": word_structure if word_structure else [len(part) for part in word.split() if part],
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
