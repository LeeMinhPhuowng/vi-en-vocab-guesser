
import sys
import os

# Thêm đường dẫn để import được ProSolverAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ProSolverAI import ProSolverAI

def manual_learn():
    ai = ProSolverAI()
    print("[MANUAL] Learning COLD-BLOODED...")
    
    # Nạp Nghĩa chính
    ai.learn_new_case(
        word="COLD-BLOODED",
        hint_vi="có máu lạnh",
        word_structure=[12],
        pos_tag="ADJECTIVE"
    )
    # Nạp Giải thích chi tiết
    ai.learn_new_case(
        word="COLD-BLOODED",
        hint_vi="Mô tả các loài động vật có nhiệt độ cơ thể thay đổi theo nhiệt độ môi trường xung quanh. Bò sát và cá là ví dụ về những loài động vật như vậy.",
        word_structure=[12],
        pos_tag="ADJECTIVE"
    )
    print("[OK] COLD-BLOODED learned successfully.")

if __name__ == "__main__":
    manual_learn()
