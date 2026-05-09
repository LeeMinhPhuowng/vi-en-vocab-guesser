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

from colorama import init, Fore, Style
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

def validate_and_clean_answer(raw_answer, expected_structure):
    if not raw_answer or not expected_structure: return None
    clean_raw = re.sub(r'[^a-zA-Z\s]', '', raw_answer).strip().lower()
    parts = clean_raw.split()
    if len(parts) == len(expected_structure):
        match = True
        for p, expected_len in zip(parts, expected_structure):
            if len(p) != expected_len:
                match = False
                break
        if match: return " ".join(parts).upper()
    if len(parts) > len(expected_structure):
        for i in range(len(parts) - len(expected_structure) + 1):
            sub_parts = parts[i : i + len(expected_structure)]
            match = True
            for p, expected_len in zip(sub_parts, expected_structure):
                if len(p) != expected_len:
                    match = False
                    break
            if match: return " ".join(sub_parts).upper()
    return None

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
            round_id = f"{letter_count}|{translation}|{ws_list!s}"
            hints_now = (str(data.get("hints") or "none")).strip()
            pos_tag, example_en_now, status = data.get("pos_tag", ""), data.get("example_en", ""), data.get("gameStatus", "playing")
            example_vi_now = data.get("example_vi", "")
            hint_vi_now = data.get("hint_vi", "")

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

            # 2. HỌC KHI KẾT THÚC (Log ngắn gọn)
            if status != "playing" or "đáp án" in current_text.lower():
                raw_revealed = str(data.get("revealedAnswer") or "").strip()
                revealed = validate_and_clean_answer(raw_revealed, ws_list)
                diff_word = extract_diff_word(last_example_en, example_en_now, expected_structure=ws_list)
                
                final_answer = revealed or diff_word
                
                if final_answer and round_id != last_learned_round_id:
                    last_learned_round_id = round_id
                    
                    # Tinh toan thoi gian giai
                    solve_time_str = "N/A"
                    if last_submit_time > 0:
                        solve_time_str = f"{int((time.perf_counter() - (last_submit_time - 0.6)) * 1000)}ms"

                    print(f"\n  {Fore.WHITE}{Style.BRIGHT}>> ĐÁP ÁN: {Fore.GREEN}{final_answer.upper()} {Fore.CYAN}({solve_time_str})")
                    msg_ans = f"  {Fore.CYAN}└─ Đáp án là: {Fore.GREEN}{final_answer.upper()}"
                    print(msg_ans); _log_to_file(msg_ans)

                    # HỌC TỪ MỚI
                    session_stats["learned"] += 1
                    threading.Thread(target=learn, args=(final_answer, current_text, translation, hint_vi_now, example_vi_now, ws_list, pos_tag), daemon=True).start()
                    
                    tried_words = []
                    time.sleep(2.0)
                    continue

            # 2. GIẢI CÂU ĐỐ (Log chi tiết để hỗ trợ làm bài)
            if letter_count > 0:
                if round_id != last_round_id:
                    tried_words, last_round_id, last_hints_logged, last_pos_logged, last_solve_input_key, last_submit_time, hints_when_submitted, last_example_en, current_hint_fails, status_logged = [], round_id, "", "", None, 0.0, "", example_en_now, 0, False
                    slots_line = format_puzzle_slots(ws_list, hints_now, letter_count)
                    
                    # IN BANG TONG HOP PREMIUM KHI MOI HIEN DE
                    print(f"\n  {Fore.CYAN}{Style.BRIGHT}[NEW QUESTION]")
                    print(f"  {Fore.CYAN}├ Đề (ô): {Style.RESET_ALL}{slots_line} (Structure: {ws_list})")
                    print(f"  {Fore.CYAN}├ Nghĩa: {Fore.WHITE}{translation or 'Nghĩa ẩn'}")
                    print(f"  {Fore.CYAN}├ Giải thích: {Fore.WHITE}{hint_vi_now or 'N/A'}")
                    print(f"  {Fore.CYAN}├ Ví dụ VN: {Fore.WHITE}{example_vi_now or 'N/A'}")
                    print(f"  {Fore.CYAN}├ Ví dụ EN: {Fore.WHITE}\"{example_en_now}\"")
                    print(f"  {Fore.CYAN}└ Loại từ: {Fore.YELLOW}{pos_tag.upper()}")

                if hints_now != last_hints_logged and last_hints_logged != "":
                    slots_line = format_puzzle_slots(ws_list, hints_now, letter_count)
                    print(f"  {Fore.MAGENTA}[DOM]{Style.RESET_ALL} Mở gợi ý → {slots_line}")
                    last_hints_logged, last_submit_time, current_hint_fails = hints_now, 0.0, 0

                solve_key = (round_id, hints_now, tuple(tried_words))
                
                # Logic delay: nếu hints không đổi (đoán sai), kiểm tra số lần sai để tăng delay
                if last_submit_time and hints_now == hints_when_submitted:
                    delay = 2.5 if current_hint_fails >= 2 else 1.5
                    if (time.perf_counter() - last_submit_time) < delay:
                        time.sleep(0.05)
                        continue
                    else:
                        # Quá thời gian mà hint không đổi -> Xác nhận đoán sai lần này
                        current_hint_fails += 1
                        last_submit_time = 0.0 # Reset để cho phép lượt đoán tiếp theo
                
                if solve_key == last_solve_input_key:
                    time.sleep(0.05)
                    continue

                t0 = time.perf_counter()
                try:
                    data["tried_words"] = tried_words 
                    answer = solve(data, config.get("model"), threshold=config.get("confidence_threshold", 0.6))
                    if not answer:
                        time.sleep(0.1)
                        continue
                    
                    msg = f"  {Fore.YELLOW}[GUESS]{Style.RESET_ALL} Đã đoán: {Fore.WHITE}{answer.upper()}"
                    print(msg)
                    _log_to_file(msg)
                    session_stats["attempts"] += 1
                    last_solve_input_key = solve_key
                    if answer not in tried_words: tried_words.append(answer)
                    
                    # Giao dien nhap lieu
                    browser.focus_input(config.get("input_selector", "input[type='text']"))
                    browser.clear_input()
                    browser.type_text(answer)
                    browser.press_enter()
                    last_submit_time, hints_when_submitted = time.perf_counter(), hints_now
                    time.sleep(0.6)
                except Exception as ai_err:
                    err(f"AI Error: {str(ai_err)}")
                    time.sleep(1)
            else:
                time.sleep(0.2)
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
