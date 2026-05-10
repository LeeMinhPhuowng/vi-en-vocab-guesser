import os
import sys
import json

# Thiết lập đường dẫn để import ProSolverAI
sys.path.insert(0, "d:/Vocab Tool")

from ProSolverAI import ai_engine

def add_word_manually(word, meaning_vi, hint_vi, pos="noun"):
    print(f"Adding '{word}' to database...")
    # Tính toán word_structure tự động
    word_structure = [len(part) for part in word.replace("-", " ").split()]
    
    # Lưu Nghĩa chính
    ai_engine.learn_new_case(word=word, hint_vi=meaning_vi, word_structure=word_structure, pos_tag=pos)
    # Lưu Giải thích chi tiết
    ai_engine.learn_new_case(word=word, hint_vi=hint_vi, word_structure=word_structure, pos_tag=pos)
    
    print(f"Successfully added '{word}'!")

if __name__ == "__main__":
    add_word_manually(
        word="light fixture",
        meaning_vi="thiết bị chiếu sáng",
        hint_vi="Một thiết bị điện chứa bóng đèn để cung cấp ánh sáng, thường được gắn cố định vào tường hoặc trần nhà.",
        pos="noun"
    )
