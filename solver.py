import re
import sys
import os
import pandas as pd

# Thiết lập đường dẫn để import ProSolverAI từ thư mục con
_SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Main Scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Import engine đã khởi tạo trong (tránh nạp model 2 lần).
from ProSolverAI import ai_engine

_solver_instance = None
_learn_cache = set()

def _get_solver():
    global _solver_instance
    if _solver_instance is None:
        _solver_instance = ai_engine
    return _solver_instance


def warmup_solver():
    """Gọi ngay khi mở tool để nạp parquet + model một lần"""
    return _get_solver()


def format_puzzle_slots(word_structure, hints_str, letter_count) -> str:
    """
    Vẽ đề dạng từng ô: [M][E][A][T] [*][*][E]... theo wordStructure + hints (1-based index).
    """
    if not word_structure or not letter_count:
        return ""
    hint_map = {}
    if hints_str and str(hints_str).lower() != "none":
        for part in re.split(r"[;,]", str(hints_str)):
            m = re.match(r"(\d+)=([a-zA-Z])", part.strip())
            if m:
                hint_map[int(m.group(1))] = m.group(2).upper()
    pos = 1
    words_out = []
    for wlen in word_structure:
        slots = []
        for _ in range(wlen):
            ch = hint_map.get(pos, "*")
            slots.append(f"[{ch}]")
            pos += 1
        words_out.append("".join(slots))
    return "  ".join(words_out)


def _extract_context(text: str):
    """
    Bóc tách thông minh dựa trên đặc điểm nội dung thay vì từ khóa cố định.
    """
    translation_vi = ""
    hint_vi = ""
    example_vi = ""
    example_en = ""

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Loại bỏ các dòng UI nhiễu để tránh lấy nhầm vào hint/example
    def _is_noise_line(s: str) -> bool:
        s0 = s.lower()
        return (
            "đấu trường từ vựng" in s0
            or "phát âm" in s0
            or "vòng tiếp theo" in s0
            or "giải trong" in s0
            or "bản dịch" in s0
            or "thách đấu" in s0
            or "nhận kim cương" in s0
            or "giây" in s0
            or "parroto" in s0
            or "học mọi lúc" in s0
        )

    lines = [l for l in lines if not _is_noise_line(l)]

    for line in lines:
        m_trans = re.match(r'^(?:Bản dịch|Ban dich)\s*:\s*(.+)$', line, re.IGNORECASE)
        if m_trans and not translation_vi:
            translation_vi = m_trans.group(1).strip()
            continue

        # 1. Nếu dòng chứa nhiều dấu sao -> Đây là câu ví dụ Tiếng Anh bị che
        if '*' in line and len(line) > 3:
            example_en = line
        
        # 2. Nếu dòng chứa ký tự tiếng Việt có dấu -> Nghĩa hoặc Ví dụ dịch
        elif re.search(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', line, re.IGNORECASE):
            # Nếu chưa có hint_vi thì dòng đầu tiên có dấu thường là nghĩa
            if not hint_vi:
                hint_vi = line  
            else:
                # Các dòng có dấu tiếp theo thường là ví dụ tiếng Việt
                example_vi = line

    return translation_vi, hint_vi, example_vi, example_en


def _build_pattern(hints_str: str, letter_count: int) -> str:
    """
    Chuyển "1=c, 3=a" thành "c.a...." để Regex quét database.
    """
    pattern = list("." * letter_count)
    if hints_str and hints_str.lower() != "none":
        # Hỗ trợ cả dấu phẩy và dấu chấm phẩy
        parts = re.split(r'[;,]', hints_str)
        for part in parts:
            m = re.match(r"(\d+)=([a-zA-Z])", part.strip())
            if m:
                idx = int(m.group(1)) - 1 # Chuyển về 0-index
                char = m.group(2).lower()
                if 0 <= idx < letter_count:
                    pattern[idx] = char
    return "".join(pattern)


def solve(game_data: dict, model_name: str = None, threshold: float = 0.5) -> str:
    solver = _get_solver()

    letter_count = game_data.get("letter_count") or game_data.get("letterCount", 0)
    hints_str    = game_data.get("hints", "none")
    tried_words  = game_data.get("tried_words", [])
    word_structure = game_data.get("wordStructure") or None
    pos_tag      = game_data.get("pos_tag", "")

    # LẤY DỮ LIỆU SẠCH TỪ BROWSER (Do JS_EXTRACT bóc tách bằng Class CSS)
    translation_vi = game_data.get("translation_vi", "")
    hint_vi = game_data.get("hint_vi", "")
    example_vi = game_data.get("example_vi", "")
    example_en = game_data.get("example_en", "")

    # CHỈ KHI NÀO JS KHÔNG LẤY ĐƯỢC (RỖNG) THÌ MỚI DÙNG FALLBACK
    if not hint_vi:
        raw_text = game_data.get("text", "")
        # Loại bỏ các dòng UI nhiễu
        lines = [
            l.strip() for l in raw_text.splitlines()
            if l.strip()
            and "Parroto" not in l
            and "học mọi lúc" not in l
            and "đấu trường từ vựng" not in l.lower()
            and "phát âm" not in l.lower()
            and "vòng tiếp theo" not in l.lower()
            and "thách đấu" not in l.lower()
            and "nhận kim cương" not in l.lower()
            and "giây" not in l.lower()
        ]
        hint_vi = lines[0] if lines else ""

    # FALLBACK: Nếu Browser không trả về đủ, mới dùng hàm bóc tách text thô
    if not hint_vi or not example_en or not translation_vi:
        raw_text = game_data.get("text", "")
        t_fallback, h_fallback, e_vi_fallback, e_en_fallback = _extract_context(raw_text)
        translation_vi = translation_vi or t_fallback
        hint_vi = hint_vi or h_fallback
        example_en = example_en or e_en_fallback
        example_vi = example_vi or e_vi_fallback

    # Ưu tiên "Bản dịch" ngắn gọn làm nghĩa chính để solve/learn ổn định.
    hint_for_solve = translation_vi or hint_vi

    pattern = _build_pattern(hints_str, letter_count)

    # GIAI ĐOẠN 1: Tìm kiếm chính xác
    results = solver.solve(
        hint_vi=hint_vi, # Bây giờ là phần Giải thích chi tiết
        length=letter_count,
        pattern=pattern,
        word_structure=word_structure,
        example_en=example_en,
        translation_vi=translation_vi, # Đây là phần Nghĩa ngắn gọn
        pos_tag=pos_tag,
        tried_words=tried_words
    )

    if results.empty:
        return ""

    # CHỌN TỪ CHƯA THỬ VÀ ĐẠT NGƯỠNG TIN CẬY
    for _, row in results.iterrows():
        word = str(row["word"]).lower().strip()
        score = row.get("final_score", 0)
        if word not in tried_words:
            if score >= threshold:
                return word
            else:
                return "" # Điểm thấp quá, không đoán

    return ""

def learn(word, game_text, translation_vi="", hint_vi="", word_structure=None, pos_tag=""):
    try:
        solver = _get_solver()

        key = word.lower().strip()
        if key in _learn_cache:
            return
        _learn_cache.add(key)

        t_fallback, h_fallback, _, _ = _extract_context(game_text)
        translation_vi = (translation_vi or t_fallback).strip()
        hint_vi = (hint_vi or h_fallback).strip()

        # Học cả bản dịch và gợi ý (nếu chúng khác nhau)
        for h in [translation_vi, hint_vi]:
            if h and len(h) > 1:
                solver.learn_new_case(
                    word=word,
                    hint_vi=h,
                    word_structure=word_structure,
                    pos_tag=pos_tag
                )
    except Exception as e:
        print(f"  [!] Lỗi Database: {e}")
