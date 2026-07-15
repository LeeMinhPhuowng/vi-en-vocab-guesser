"""
Configuration management for Vocab Solver.
"""
import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULT_CONFIG = {
    "cdp_port": 9222,
    "input_selector": "input[data-slot='input']",
    "submit_selector": "form", 
    "hotkey_solve": "`",
    "hotkey_quit": "esc",
    "model": "distiluse-base-multilingual-cased-v2", 
    "debug_mode": False,
    # "loop_delay": 3.0,  # Khong con su dung, thay bang delay thong minh trong main.py
    "confidence_threshold": 4.0
}


def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config.update(json.load(f))
        except (json.JSONDecodeError, IOError):
            pass
    return config


def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
