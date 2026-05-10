"""
Vocab Guesser - Real-time vocabulary game auto-solver.
Reads DOM via Chrome DevTools Protocol, solves with Gemini, types answer.
"""
import os
import sys
import io
import time
import threading
import traceback
import re

import solver

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from colorama import init, Fore, Style, Back
import keyboard

from config import load_config
from browser import Browser
from solver import solve, learn, warmup_solver, format_puzzle_slots

init(autoreset=True)

BANNER = f"""
{Fore.CYAN}+==================================================+
|  {Fore.GREEN}VOCAB Auto Guesser{Fore.CYAN}          |
+==================================================+
|  {Fore.YELLOW}F9 {Fore.WHITE} ->  Solve (doan + nhap dap an)  ~0.6s     {Fore.CYAN}|
|  {Fore.YELLOW}F10{Fore.WHITE} ->  Exit                                  {Fore.CYAN}|
+==================================================+{Style.RESET_ALL}
"""

lock = threading.Lock()
browser = None
auto_mode = False
tried_words = []
last_learned_round_id = ""
last_round_id = ""
last_hints_logged = ""
last_pos_logged = ""
last_example_en = ""
last_solve_input_key = None
last_submit_time = 0.0
hints_when_submitted = ""
current_hint_fails = 0
status_logged = False


# Thong ke session
session_stats = {"attempts": 0, "correct": 0, "not_found": 0, "learned": 0}

def _log_to_file(msg):
    clean_msg = re.sub(r'\x1b\[[0-9;]*m', '', msg)
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(clean_msg + "\n")

def info(msg):
    out = f"  {Fore.CYAN}[>]{Style.RESET_ALL} {msg}"
    print(out)
    _log_to_file(out)

def ok(msg):
    out = f"  {Fore.GREEN}[OK]{Style.RESET_ALL} {msg}"
    print(out)
    _log_to_file(out)

def warn(msg):
    out = f"  {Fore.YELLOW}[!]{Style.RESET_ALL} {msg}"
    print(out)
    _log_to_file(out)

def err(msg):
    out = f"  {Fore.RED}[X]{Style.RESET_ALL} {msg}"
    print(out)
    _log_to_file(out)

def finish(msg):
    out = f"  {Fore.MAGENTA}[FINISH]{Style.RESET_ALL} {msg}"
    print(out)
    _log_to_file(out)

def validate_and_clean_answer(raw_answer, expected_structure=None):
    if not raw_answer: return None
    # Lấy Y ĐÚC những gì web hiện ra, chỉ bỏ khoảng trắng thừa ở đầu/cuối
    return raw_answer.strip().upper()

def extract_diff_word(old_ex, new_ex, expected_structure=None):
    if not old_ex or not new_ex or old_ex == new_ex: return None
    o_words, n_words = old_ex.split(), new_ex.split()
    if len(o_words) != len(n_words): return None
    diff_parts = []
    for o, n in zip(o_words, n_words):
        if '*' in o or '.' in o:
            clean_n = re.sub(r'[^a-zA-Z0-9\s]', '', n)
            if clean_n: diff_parts.append(clean_n)
    raw_res = " ".join(diff_parts).strip()
    return validate_and_clean_answer(raw_res, expected_structure)

def connect_browser(config):
    global browser
    if browser is None: browser = Browser(port=config.get("cd_port", 9222))
    if browser.connect():
        ok("Đã kết nối trình duyệt.")
        return True
    return False

def auto_solve_loop(config):
    global auto_mode, browser, tried_words, last_learned_round_id, last_round_id, last_hints_logged, last_pos_logged, last_solve_input_key, last_submit_time, hints_when_submitted, last_example_en
    last_valid_text = ""; last_valid_translation = ""; last_valid_hint_vi = ""
    last_valid_pos = ""; last_valid_ws = []
    status_logged = False # Khởi tạo để tránh lỗi UnboundLocalError

    while auto_mode:
        if not lock.acquire(blocking=False):
            time.sleep(0.05)
            continue
            
        try:
            if browser is None or browser.ws is None:
                if not connect_browser(config):
                    time.sleep(1.5)
                    continue
            
            data = browser.read_game_data()
            if data.get("status") == "disconnected":
                time.sleep(1)
                continue
            
            current_text, letter_count = data.get("text", ""), data.get("letterCount", 0)
            translation, ws_list = (data.get("translation_vi") or "").strip(), data.get("wordStructure") or []
            if ws_list and sum(ws_list) > 0:
                auto_solve_loop.last_valid_ws = ws_list
            
            round_id = f"{letter_count}|{translation}|{ws_list!s}"
            hints_now = (str(data.get("hints") or "none")).strip()
            pos_tag, example_en_now, status = data.get("pos_tag", ""), data.get("example_en", ""), data.get("gameStatus", "playing")
            hint_vi_now = data.get("hint_vi", "")

            # 0. LƯU NGỮ CẢNH (Để học sau khi round kết thúc)
            if status == "playing":
                last_valid_text = data.get("text", "")
                last_valid_translation = data.get("translation_vi", "")
                last_valid_hint_vi = data.get("hint_vi", "")
                last_valid_pos = data.get("pos_tag", "")
                last_valid_ws = ws_list

            # 1. LOG TRẠNG THÁI (Ngay khi kết thúc câu hỏi)
            if status != "playing" and not status_logged:
                status_logged = True
                
                if status == "player_correct":
                    msg = f"  {Fore.GREEN}[OK] Bạn đã đoán đúng"
                    print(msg); _log_to_file(msg)
                    session_stats["correct"] += 1

                elif status == "enemy_correct":
                    msg = f"  {Fore.YELLOW}[!] Đối thủ đã đoán đúng"
                    print(msg); _log_to_file(msg)
                elif status == "timeout":
                    msg = f"  {Fore.MAGENTA}[TIMEOUT] Hết giờ"
                    print(msg); _log_to_file(msg)

            # 2. HỌC KHI KẾT THÚC (Dành cho trường hợp hết giờ hoặc đối thủ đoán)
            if status != "playing" or "đáp án" in current_text.lower():
                final_answer = None
                raw_revealed = ""
                # SỬ DỤNG CẤU TRÚC ĐÃ LƯU (Tránh việc ô chữ biến mất làm rỗng ws_list)
                current_ws = ws_list if (ws_list and sum(ws_list) > 0) else getattr(auto_solve_loop, "last_valid_ws", [])
                
                if current_ws and sum(current_ws) > 0:
                    for _ in range(12): # Tang len 12 lan de khong bo lo
                        raw_revealed = str(data.get("revealedAnswer") or "").strip()
                        revealed = validate_and_clean_answer(raw_revealed, current_ws)
                        if revealed:
                            final_answer = revealed
                            break
                        time.sleep(0.25) # Poll nhanh hon
                        data = browser.read_game_data()

                # FALLBACK: Nếu bạn đoán đúng nhưng không quét được màn hình, lấy từ tried_words
                if not final_answer and status == "player_correct" and tried_words:
                    final_answer = tried_words[-1]
                if final_answer and round_id != getattr(auto_solve_loop, "last_logged_ans_id", ""):
                    auto_solve_loop.last_logged_ans_id = round_id
                    print(f"\n  {Back.GREEN}{Fore.BLACK} ĐÁP ÁN THẬT: {final_answer.upper()} {Style.RESET_ALL}")
                    msg_ans = f"  {Fore.CYAN}└─ Đáp án của câu hỏi là: {Fore.GREEN}{final_answer.upper()}"
                    _log_to_file(msg_ans)
                # HỌC TỪ MỚI (Sử dụng ngữ cảnh đã bảo lưu)
                if final_answer and round_id != last_learned_round_id:
                    last_learned_round_id = round_id
                    session_stats["learned"] += 1
                    print(f"  {Fore.BLUE}[LEARN]{Style.RESET_ALL} Đang lưu kiến thức: {Fore.WHITE}{final_answer.upper()} -> {last_valid_translation}")
                    # Sử dụng dữ liệu đã bảo lưu để học chính xác
                    threading.Thread(
                        target=learn, 
                        args=(final_answer, last_valid_text, last_valid_translation, last_valid_hint_vi, last_valid_ws, last_valid_pos), 
                        daemon=True
                    ).start()
                    
                    tried_words = []
                    time.sleep(1.0)
                    continue
                elif status != "playing":
                    # Đợi thêm một chút để đáp án kịp hiện ra (Đặc biệt quan trọng khi đối thủ thắng)
                    if status == "enemy_correct":
                        time.sleep(1.2)
                        # Thử bắt lại đáp án một lần nữa
                        data_retry = browser.read_game_data()
                        final_answer = data_retry.get("revealedAnswer")
                    
                    tried_words = []
                    time.sleep(0.5)
                    continue

            # 3. GIẢI CÂU ĐỐ (Chỉ thực hiện khi game ĐANG CHƠI)
            if status != "playing":
                # Nếu không phải trạng thái chơi, bỏ qua toàn bộ phần giải đố bên dưới
                time.sleep(0.5)
                continue

            if letter_count > 0:
                if round_id != last_round_id:
                    tried_words, last_round_id, last_hints_logged, last_pos_logged, last_solve_input_key, last_submit_time, hints_when_submitted, last_example_en, current_hint_fails, status_logged = [], round_id, "", "", None, 0.0, "", example_en_now, 0, False
                    slots_line = format_puzzle_slots(ws_list, hints_now, letter_count)
                    
                    # IN BANG TONG HOP PREMIUM KHI MOI HIEN DE
                    print(f"\n  {Fore.CYAN}{Style.BRIGHT}[NEW QUESTION]")
                    print(f"  {Fore.CYAN}├ Đề (ô): {Style.RESET_ALL}{slots_line} (Structure: {ws_list})")
                    print(f"  {Fore.CYAN}├ Nghĩa: {Fore.WHITE}{translation or 'Nghĩa ẩn'}")
                    print(f"  {Fore.CYAN}├ Giải thích: {Fore.WHITE}{hint_vi_now or 'N/A'}")
                    print(f"  {Fore.CYAN}├ Ví dụ EN: {Fore.WHITE}\"{example_en_now}\"")
                    print(f"  {Fore.CYAN}└ Loại từ: {Fore.YELLOW}{pos_tag.upper()}")

                if hints_now != last_hints_logged and last_hints_logged != "":
                    slots_line = format_puzzle_slots(ws_list, hints_now, letter_count)
                    print(f"  {Fore.MAGENTA}[DOM]{Style.RESET_ALL} Mở gợi ý → {slots_line}")
                    last_hints_logged, last_submit_time, current_hint_fails = hints_now, 0.0, 0

                solve_key = (round_id, hints_now, tuple(tried_words))
                
                # Logic delay: nếu hints không đổi (đoán sai), kiểm tra số lần sai để tăng delay
                if last_submit_time and hints_now == hints_when_submitted:
                    delay = 5.0 if current_hint_fails >= 2 else 3.0
                    if (time.perf_counter() - last_submit_time) < delay:
                        time.sleep(0.2)
                        continue
                    else:
                        # Quá thời gian mà hint không đổi -> Xác nhận đoán sai lần này
                        current_hint_fails += 1
                        last_submit_time = 0.0 # Reset để cho phép lượt đoán tiếp theo
                
                if solve_key == last_solve_input_key:
                    time.sleep(0.02) # Phản hồi nhanh hơn
                    continue

                t0 = time.perf_counter()
                try:
                    data["tried_words"] = tried_words 
                    answer = solve(data, config.get("model"), threshold=config.get("confidence_threshold", 0.6))
                    if not answer:
                        last_solve_input_key = solve_key
                        time.sleep(0.1)
                        continue
                    
                    msg = f"  {Fore.YELLOW}[GUESS]{Style.RESET_ALL} Đã đoán: {Fore.WHITE}{answer.upper()}"
                    print(msg)
                    _log_to_file(msg)
                    session_stats["attempts"] += 1
                    last_solve_input_key = solve_key
                    answer = answer.lower()
                    if "-" in answer and ws_list and len(ws_list) > 1:
                        answer = answer.replace("-", " ")
                    
                    if answer not in tried_words: tried_words.append(answer)
                    
                    browser.focus_input("input")
                    browser.clear_input()
                    time.sleep(0.02) # Chờ cực ngắn
                    browser.type_text(answer)
                    browser.press_enter()
                    last_submit_time, hints_when_submitted = time.perf_counter(), hints_now
                    time.sleep(0.1) # Nghỉ cực ngắn sau khi submit
                except Exception as ai_err:
                    err(f"AI Error: {str(ai_err)}")
                    time.sleep(0.5)
            else:
                time.sleep(0.05)
        except Exception as e:
            err(f"Loop Error: {str(e)}")
            time.sleep(1)
        finally:
            if lock.locked(): lock.release()
        time.sleep(0.05)

def toggle_auto(config):
    global auto_mode, last_round_id, last_hints_logged, last_pos_logged, tried_words
    auto_mode = not auto_mode
    if auto_mode:
        last_round_id, last_hints_logged, last_pos_logged, tried_words = "", "", "", []
        ok("AUTO MODE: ON.")
        threading.Thread(target=auto_solve_loop, args=(config,), daemon=True).start()
    else:
        warn("AUTO MODE: OFF.")
        log_stats("[AUTO MODE: OFF]", solver.ai_engine.get_db_size())

def log_stats(prefix="[STATS]", db_end=None):
    acc = (session_stats['correct'] / session_stats['attempts'] * 100) if session_stats['attempts'] > 0 else 0
    summary = f"\n{'='*50}\n  {prefix}\n  Lần thử:   {session_stats['attempts']}\n  Đúng:      {session_stats['correct']}  ({acc:.1f}%)\n  Đã học:    {session_stats['learned']} từ mới\n  DB size:   {db_end if db_end else 'N/A'}\n{'='*50}"
    print(f"\n{Fore.CYAN}{summary}{Style.RESET_ALL}")
    _log_to_file(summary)

def main():
    print(BANNER)
    config = load_config()
    connect_browser(config)
    warmup_solver()
    keyboard.add_hotkey(config.get("hotkey_solve", "f9"), lambda: toggle_auto(config))
    keyboard.wait(config.get("hotkey_quit", "f10"))
    if browser: browser.close()
    log_stats("[EXIT]", solver.ai_engine.get_db_size())

if __name__ == "__main__": main()
