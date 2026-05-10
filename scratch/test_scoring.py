
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Simulate the problem
device = 'cpu'
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2').to(device)

def test_scoring():
    query_trans = "người đầu tư"
    query_hint = "Đây là một người hoặc một nhóm người bỏ tiền vào một công ty hoặc dự án với hy vọng kiếm được nhiều tiền hơn sau này. Họ chấp nhận rủi ro để đạt được lợi ích tài chính trong tương lai."
    query_en = "A new ******** joined the company board.".replace("*", " ")
    
    # Candidates
    candidates = [
        {"word": "investor", "meaning": "nhà đầu tư, người đầu tư"},
        {"word": "employee", "meaning": "nhân viên, người lao động"},
        {"word": "entrance", "meaning": "lối vào, sự đi vào, cổng"},
        {"word": "interest", "meaning": "sự quan tâm, lãi suất, quyền lợi"}
    ]
    
    trans_emb = model.encode(query_trans, convert_to_tensor=True)
    hint_emb = model.encode(query_hint, convert_to_tensor=True)
    en_emb = model.encode(query_en, convert_to_tensor=True)
    
    cand_meanings = [c["meaning"] for c in candidates]
    cand_embs = model.encode(cand_meanings, convert_to_tensor=True)
    
    cand_words = [c["word"] for c in candidates]
    word_embs = model.encode(cand_words, convert_to_tensor=True)
    
    s_trans = util.cos_sim(trans_emb, cand_embs)[0]
    s_hint = util.cos_sim(hint_emb, cand_embs)[0]
    s_en_new = util.cos_sim(en_emb, word_embs)[0] # New logic: Word vs Example
    
    print(f"Query Trans: {query_trans}")
    print("-" * 50)
    for i, c in enumerate(candidates):
        score_trans = s_trans[i].item()
        score_hint = s_hint[i].item()
        score_en = s_en_new[i].item()
        
        # Base Score (60/20/20)
        final = 0.60 * score_trans + 0.20 * score_hint + 0.20 * score_en
        
        # Bonuses
        t_low = query_trans.lower()
        if t_low in c["meaning"].lower():
            final += 1.5
            
        db_meanings = [m.strip().lower() for m in c["meaning"].split(",")]
        for m in db_meanings:
            if len(m) > 2 and m in query_hint.lower():
                final += 1.0
                break
        
        print(f"Word: {c['word']:10} | Meaning: {c['meaning']:30}")
        print(f"  s_trans: {score_trans:.4f} | s_hint: {score_hint:.4f} | s_en: {score_en:.4f} | Final: {final:.4f}")

    # Now test the "Better English" theory
    print("\n--- Testing Better English (Word vs Example) ---")
    cand_words = [c["word"] for c in candidates]
    word_embs = model.encode(cand_words, convert_to_tensor=True)
    s_en_better = util.cos_sim(en_emb, word_embs)[0]
    for i, c in enumerate(candidates):
        print(f"Word: {c['word']:10} | s_en_better: {s_en_better[i].item():.4f}")

if __name__ == "__main__":
    test_scoring()
