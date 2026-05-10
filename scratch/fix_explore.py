
import sys
import os
import re
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ProSolverAI import ProSolverAI

def fix_explore():
    ai = ProSolverAI()
    print("[MANUAL] Fixing EXPLORE ALL AVENUES...")
    
    # 1. Xóa bản lỗi (không dấu cách) nếu có
    ai.vocab_db = ai.vocab_db[ai.vocab_db['word'] != 'exploreallavenues']
    
    # 2. Nạp bản chuẩn (có dấu cách)
    ai.learn_new_case(
        word="EXPLORE ALL AVENUES",
        hint_vi="tìm mọi cách",
        word_structure=[7, 3, 7],
        pos_tag="IDIOM"
    )
    ai.learn_new_case(
        word="EXPLORE ALL AVENUES",
        hint_vi="Điều này có nghĩa là thử mọi phương pháp hoặc lựa chọn có thể để tìm ra giải pháp hoặc đạt được mục tiêu. Bạn xem xét mọi khả năng trước khi đưa ra quyết định.",
        word_structure=[7, 3, 7],
        pos_tag="IDIOM"
    )
    print("[OK] EXPLORE ALL AVENUES fixed successfully.")

if __name__ == "__main__":
    fix_explore()
