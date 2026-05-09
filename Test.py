
import sys
import os
import numpy as np
import pandas as pd

# =========================================================
# IMPORT
# =========================================================

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Main Scripts"
    )
)

from ProSolverAI import ProSolverAI


# =========================================================
# BENCHMARK DATA
# =========================================================

GAME_BENCHMARK_V3 = [

    # =====================================================
    # BASIC NOUNS
    # =====================================================

    {
        "hint_vi": "máy tính",
        "example_vi": "Tôi sử dụng máy tính để làm việc.",
        "answer": "computer"
    },

    {
        "hint_vi": "bác sĩ",
        "example_vi": "Bác sĩ đang khám bệnh.",
        "answer": "doctor"
    },

    {
        "hint_vi": "thành phố",
        "example_vi": "Hà Nội là thành phố lớn.",
        "answer": "city"
    },

    {
        "hint_vi": "thư viện",
        "example_vi": "Sinh viên đến thư viện học bài.",
        "answer": "library"
    },

    # =====================================================
    # VERBS
    # =====================================================

    {
        "hint_vi": "xây dựng",
        "example_vi": "Họ đang xây dựng cây cầu.",
        "answer": "build"
    },

    {
        "hint_vi": "phát triển",
        "example_vi": "Công ty phát triển nhanh.",
        "answer": "develop"
    },

    {
        "hint_vi": "giải quyết",
        "example_vi": "Chúng ta cần giải quyết vấn đề.",
        "answer": "solve"
    },

    {
        "hint_vi": "chuẩn bị",
        "example_vi": "Mẹ đang chuẩn bị bữa tối.",
        "answer": "prepare"
    },

    # =====================================================
    # DIRECT ACTIONS
    # =====================================================

    {
        "hint_vi": "bắt",
        "example_vi": "Cậu bé bắt quả bóng.",
        "answer": "catch"
    },

    {
        "hint_vi": "nhảy",
        "example_vi": "Con ếch nhảy trên lá.",
        "answer": "jump"
    },

    {
        "hint_vi": "khóc",
        "example_vi": "Em bé đang khóc.",
        "answer": "cry"
    },

    {
        "hint_vi": "cười",
        "example_vi": "Mọi người đều cười.",
        "answer": "laugh"
    },

    # =====================================================
    # PHRASAL VERBS
    # =====================================================

    {
        "hint_vi": "tìm kiếm",
        "example_vi": "Tôi đang tìm kiếm chìa khóa.",
        "answer": "look for"
    },

    {
        "hint_vi": "thức dậy",
        "example_vi": "Tôi thức dậy lúc sáu giờ.",
        "answer": "wake up"
    },

    {
        "hint_vi": "tắt",
        "example_vi": "Hãy tắt đèn.",
        "answer": "turn off"
    },

    {
        "hint_vi": "bật",
        "example_vi": "Bạn có thể bật TV không?",
        "answer": "turn on"
    },

    # =====================================================
    # HYPHEN WORDS
    # =====================================================

    {
        "hint_vi": "đã qua sử dụng",
        "example_vi": "Tôi mua xe đã qua sử dụng.",
        "answer": "second-hand"
    },

    {
        "hint_vi": "toàn thời gian",
        "example_vi": "Cô ấy tìm việc toàn thời gian.",
        "answer": "full-time"
    },

    {
        "hint_vi": "ngắn hạn",
        "example_vi": "Đây là giải pháp ngắn hạn.",
        "answer": "short-term"
    },

    {
        "hint_vi": "tự tin",
        "example_vi": "Bạn cần tự tin hơn.",
        "answer": "self-confident"
    },

    # =====================================================
    # LONG WORDS
    # =====================================================

    {
        "hint_vi": "môi trường",
        "example_vi": "Chúng ta cần bảo vệ môi trường.",
        "answer": "environment"
    },

    {
        "hint_vi": "chính phủ",
        "example_vi": "Chính phủ ban hành luật mới.",
        "answer": "government"
    },

    {
        "hint_vi": "công nghệ",
        "example_vi": "Công nghệ thay đổi thế giới.",
        "answer": "technology"
    },

    {
        "hint_vi": "giáo dục",
        "example_vi": "Giáo dục rất quan trọng.",
        "answer": "education"
    }
]


# =========================================================
# HELPERS
# =========================================================

def normalize_word(word):

    return (
        str(word)
        .lower()
        .replace(" ", "")
        .strip()
    )


# =========================================================
# RUN TEST
# =========================================================

def run_test(solver):

    results = []

    print(f"\n{'='*80}")
    print(f"🔥 PROSOLVERAI V3 BENCHMARK")
    print(f"📊 TOTAL SAMPLES: {len(GAME_BENCHMARK_V3)}")
    print(f"{'='*80}")

    print(
        f"{'ID':<4}"
        f"{'TRUE':<22}"
        f"{'PREDICT':<22}"
        f"{'RESULT'}"
    )

    print("-" * 80)

    for i, sample in enumerate(GAME_BENCHMARK_V3):

        true_word = sample['answer']

        # normalize như game
        normalized = normalize_word(true_word)

        char_length = len(normalized)

        # pattern giả lập:
        # reveal chữ đầu + chữ cuối
        if char_length >= 2:

            pattern = (
                normalized[0]
                + "." * (char_length - 2)
                + normalized[-1]
            )

        else:
            pattern = normalized

        try:
            word_structure = [ len(x) for x in true_word.split() ]

            pred_df = solver.solve(
                hint_vi=sample['hint_vi'],
                length=char_length,
                pattern=pattern,
                word_structure=word_structure,
                example_en="",
                example_vi=sample['example_vi']
            )

            if pred_df.empty:
                pred_word = "(not found)"
            else:
                pred_word = str(
                    pred_df.iloc[0]['word']
                ).lower().strip()

        except Exception as e:

            pred_word = f"(error: {e})"

        is_correct = (
            normalize_word(pred_word)
            ==
            normalize_word(true_word)
        )

        results.append(is_correct)

        status = "✅" if is_correct else "❌"

        print(
            f"{i+1:<4}"
            f"{true_word:<22}"
            f"{pred_word:<22}"
            f"{status}"
        )

    print("-" * 80)

    correct = sum(results)

    total = len(results)

    acc = np.mean(results) * 100

    print(f"CORRECT : {correct}/{total}")
    print(f"ACCURACY: {acc:.2f}%")

    print(f"{'='*80}\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    # data.parquet cùng cấp file benchmark
    db = os.path.join(
        current_dir,
        "data.parquet"
    )

    print(f"[INFO] DB PATH: {db}")

    if not os.path.exists(db):

        print(f"[ERROR] Khong tim thay DB")

        print(
            "Files:",
            os.listdir(current_dir)
        )

    else:

        print("[INFO] Dang khoi dong AI Engine...")

        solver = ProSolverAI(
            db_path=db
        )

        run_test(solver)

