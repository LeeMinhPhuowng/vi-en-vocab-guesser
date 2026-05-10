
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Sử dụng model từ config
device = 'cpu'
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2').to(device)

def test_words(test_cases):
    print(f"{'Word':<15} | {'Status':<10} | {'Score':<8} | {'s_trans':<8} | {'s_hint':<8} | {'s_en':<8}")
    print("-" * 75)
    
    for case in test_cases:
        query_trans = case["q_trans"]
        query_hint = case["q_hint"]
        query_en = case["q_en"].replace("*", " ")
        expected = case["expected"]
        
        # Encode inputs
        trans_emb = model.encode(query_trans, convert_to_tensor=True)
        hint_emb = model.encode(query_hint, convert_to_tensor=True)
        en_emb = model.encode(query_en, convert_to_tensor=True)
        
        # Simulate candidates (Expected + 2 Distractors)
        candidates = [
            {"word": expected, "meaning": case["db_meaning"]},
            {"word": case["dist1"]["word"], "meaning": case["dist1"]["meaning"]},
            {"word": case["dist2"]["word"], "meaning": case["dist2"]["meaning"]},
        ]
        
        results = []
        for cand in candidates:
            cand_meaning_emb = model.encode(cand["meaning"], convert_to_tensor=True)
            cand_word_emb = model.encode(cand["word"], convert_to_tensor=True)
            
            s_trans = util.cos_sim(trans_emb, cand_meaning_emb)[0].item()
            s_hint = util.cos_sim(hint_emb, cand_meaning_emb)[0].item()
            s_en = util.cos_sim(en_emb, cand_word_emb)[0].item()
            
            # Final Score Logic (60/20/20)
            score = 0.60 * s_trans + 0.20 * s_hint + 0.20 * s_en
            
            # Bonuses
            if query_trans.lower() in cand["meaning"].lower():
                score += 1.5
            
            db_hints = [h.strip().lower() for h in cand["meaning"].split(",") if len(h.strip()) > 2]
            for dh in db_hints:
                if dh in query_hint.lower():
                    score += 1.0
                    break
            
            results.append({"word": cand["word"], "score": score, "s_trans": s_trans, "s_hint": s_hint, "s_en": s_en})
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        winner = results[0]
        status = "✅ PASS" if winner["word"].lower() == expected.lower() else "❌ FAIL"
        
        print(f"{expected:<15} | {status:<10} | {winner['score']:<8.4f} | {winner['s_trans']:<8.4f} | {winner['s_hint']:<8.4f} | {winner['s_en']:<8.4f}")
        if status == "❌ FAIL":
            print(f"   [!] Winner was: {winner['word']} instead of {expected}")

if __name__ == "__main__":
    test_cases = [
        {
            "q_trans": "người đầu tư",
            "q_hint": "Một người bỏ tiền vào công ty để kiếm lời.",
            "q_en": "A new ******** joined the board.",
            "expected": "investor",
            "db_meaning": "nhà đầu tư, người đầu tư",
            "dist1": {"word": "employee", "meaning": "nhân viên"},
            "dist2": {"word": "entrance", "meaning": "lối vào"}
        },
        {
            "q_trans": "xuất sắc",
            "q_hint": "Cực kỳ tốt, vượt xa mức trung bình.",
            "q_en": "She gave an *********** performance.",
            "expected": "outstanding",
            "db_meaning": "xuất sắc, nổi bật, đáng chú ý",
            "dist1": {"word": "interesting", "meaning": "thú vị"},
            "dist2": {"word": "overarching", "meaning": "vòm, bao quát"}
        },
        {
            "q_trans": "chứng ám ảnh",
            "q_hint": "Nỗi sợ hãi vô lý đối với cái gì đó.",
            "q_en": "He has a ****** of spiders.",
            "expected": "phobia",
            "db_meaning": "ám ảnh, chứng sợ hãi",
            "dist1": {"word": "photos", "meaning": "ảnh"},
            "dist2": {"word": "phones", "meaning": "điện thoại"}
        },
        {
            "q_trans": "bài tập về nhà",
            "q_hint": "Công việc giáo viên giao về nhà.",
            "q_en": "I must finish my ********.",
            "expected": "homework",
            "db_meaning": "bài tập về nhà, việc nhà",
            "dist1": {"word": "homeland", "meaning": "quê hương"},
            "dist2": {"word": "homemade", "meaning": "tự làm"}
        },
        {
            "q_trans": "vắng mặt",
            "q_hint": "Không có mặt ở nơi cần thiết.",
            "q_en": "The student was ****** from class.",
            "expected": "absent",
            "db_meaning": "vắng mặt, nghỉ học",
            "dist1": {"word": "absorb", "meaning": "hấp thụ"},
            "dist2": {"word": "accept", "meaning": "chấp nhận"}
        },
        {
            "q_trans": "giai điệu",
            "q_hint": "Một chuỗi các nốt nhạc dễ nhớ.",
            "q_en": "He hummed a happy ****.",
            "expected": "tune",
            "db_meaning": "giai điệu, bản nhạc",
            "dist1": {"word": "time", "meaning": "thời gian"},
            "dist2": {"word": "type", "meaning": "kiểu"}
        },
        {
            "q_trans": "cài đặt",
            "q_hint": "Các cấu hình kiểm soát chương trình phần mềm. Cho phép tùy chỉnh hệ thống.",
            "q_en": "Go to ******** to change your wallpaper.",
            "expected": "settings",
            "db_meaning": "cài đặt, thiết lập, tùy chọn",
            "dist1": {"word": "software", "meaning": "phần mềm"},
            "dist2": {"word": "strategy", "meaning": "chiến lược"}
        }
    ]
    test_words(test_cases)
