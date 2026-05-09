"""
Chrome DevTools Protocol (CDP) connector.
Reads game DOM and types answers via trusted keyboard events.
"""
import json
import requests
import websocket
import time


# JS snippet to extract game data from the page
JS_EXTRACT = r"""(() => {
    try {
        const mainContent = document.querySelector('main') || document.body;

        let translation_vi = "";
        let hint_vi = "";
        let example_vi = "";
        let example_en = "";
        let pos_tag = "";

        const normalize = (s) => (s || "")
            .toLowerCase()
            .normalize("NFD")
            .replace(/[\u0300-\u036f]/g, "")
            .replace(/đ/g, "d")
            .replace(/[:：]/g, "")
            .replace(/\s+/g, " ")
            .trim();

        const labels = [...mainContent.querySelectorAll("span.font-semibold")];
        
        for (const labelNode of labels) {
            const label = normalize(labelNode.innerText);
            if (!label) continue;

            let valueNode = null;
            const parent = labelNode.parentElement;
            if (parent) {
                const candidates = [...parent.querySelectorAll("span")].filter((el) => (
                    el !== labelNode &&
                    !el.classList.contains("font-semibold")
                ));
                valueNode = candidates.find((el) => !!el.innerText && el.innerText.trim()) || null;
            }
            const value = valueNode ? valueNode.innerText.replace(/\n/g, " ").trim() : "";
            if (!value) continue;

            if (!translation_vi && (label.includes("ban dich") || label.includes("translation"))) translation_vi = value;
            else if (!hint_vi && (label.includes("giai thich") || label.includes("nghia"))) hint_vi = value;
            else if (!example_vi && label.includes("vi du") && !label.includes("tieng anh")) {
                // Kiểm tra nếu ví dụ VN chứa quá nhiều tiếng Anh hoặc dấu * thì bỏ qua
                const hasVietnamese = /[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]/i.test(value);
                if (hasVietnamese) example_vi = value;
            }
            else if (!example_en && (label.includes("vi du tieng anh") || label.includes("example"))) example_en = value;
            else if (!pos_tag && (label.includes("loai tu") || label.includes("part of speech") || label.includes("pos"))) pos_tag = value;
        }

        if (!example_en) {
            const exEnEl = mainContent.querySelector("span.italic");
            example_en = exEnEl ? exEnEl.innerText.replace(/\n/g, " ").trim() : "";
        }

        const bodyText = mainContent.innerText || "";
        
        // =====================================================
        // WORD STRUCTURE DETECTION (FINE-TUNED GAP)
        // =====================================================
        const allSpans = [...mainContent.querySelectorAll('span[class*="w-7"][class*="h-9"]')]
            .filter((el) => {
                const t = (el.innerText || "").trim();
                return t === "*" || (t.length === 1 && /[a-zA-Z]/.test(t));
            });
        
        let rows = [];
        allSpans.forEach(el => {
            const rect = el.getBoundingClientRect();
            let found = false;
            for (const row of rows) {
                if (Math.abs(row.y - rect.y) < 12) { row.items.push(el); found = true; break; }
            }
            if (!found) rows.push({ y: rect.y, items: [el] });
        });

        rows.sort((a,b)=>a.y-b.y);
        const wordStructure = [];
        const hints = [];
        const revealedChars = {};
        let totalLetters = 0;
        let globalIndex = 0;

        rows.forEach(row => {
            const letters = row.items
                .map((box) => ({ box, rect: box.getBoundingClientRect() }))
                .sort((a, b) => a.rect.x - b.rect.x);
            
            if (letters.length === 0) return;

            const widths = letters.map(l => l.rect.width).filter(w => w > 0);
            const avgW = widths.length ? (widths.reduce((a,b)=>a+b,0)/widths.length) : 28;
            // Dùng 0.8 làm điểm cân bằng: Không gộp nhầm từ ngắn, không tách nhầm từ dài.
            const splitGap = avgW * 0.8; 

            let currentWordLen = 0;
            for (let i = 0; i < letters.length; i++) {
                const item = letters[i];
                const t = item.box.innerText.trim();
                totalLetters++;
                currentWordLen++;
                if (t && t.length === 1 && t !== "*") {
                    hints.push(`${globalIndex + 1}=${t}`);
                    revealedChars[globalIndex + 1] = t.toUpperCase();
                }
                globalIndex++;

                const next = letters[i+1];
                if (!next) {
                    if (currentWordLen > 0) wordStructure.push(currentWordLen);
                } else {
                    const gap = next.rect.x - (item.rect.x + item.rect.width);
                    if (gap > splitGap) {
                        if (currentWordLen > 0) wordStructure.push(currentWordLen);
                        currentWordLen = 0;
                    }
                }
            }
        });

        let gameStatus = "playing";
        if (bodyText.includes("Bạn đã đoán đúng") || bodyText.includes("Chúc mừng")) gameStatus = "player_correct";
        else if (bodyText.includes("Đối thủ đã đoán đúng") || bodyText.includes("Rất tiếc")) gameStatus = "enemy_correct";
        else if (bodyText.includes("Hết giờ")) gameStatus = "timeout";

        let revealedAnswer = "";
        if (gameStatus !== "playing") {
            let parts = [];
            let cIdx = 1;
            wordStructure.forEach(wlen => {
                let w = "";
                for (let i=0; i<wlen; i++) { 
                    let ch = revealedChars[cIdx];
                    if (ch) w += ch;
                    cIdx++; 
                }
                if (w.length === wlen) parts.push(w);
            });
            if (parts.length > 0 && parts.length === wordStructure.length) revealedAnswer = parts.join(" ");
        }

        if (!revealedAnswer || revealedAnswer.includes("*")) {
            const resultSpans = [...mainContent.querySelectorAll("span.text-green-600, span.text-green-700, .font-bold.text-2xl")].filter(s => {
                const text = s.innerText.trim();
                return /^[a-zA-Z\s\-]+$/.test(text) && text.replace(/\s+/g, "").length >= 2;
            });
            const match = resultSpans.find(s => s.innerText.replace(/[^a-zA-Z]/g, "").length === totalLetters);
            if (match) revealedAnswer = match.innerText.trim().toUpperCase();
        }

        return JSON.stringify({
            status: "ok",
            text: bodyText.substring(0, 3000),
            letterCount: totalLetters,
            wordStructure: wordStructure,
            hints: hints.join(", "),
            gameStatus: gameStatus,
            revealedAnswer: revealedAnswer,
            translation_vi: translation_vi,
            hint_vi: hint_vi,
            example_en: example_en,
            example_vi: example_vi,
            pos_tag: pos_tag
        });

    } catch(err) { return JSON.stringify({ status: "error", message: err.message }); }
})()"""


class Browser:
    def __init__(self, port=9222):
        self.port = port
        self.ws = None
        self._id = 0

    def connect(self):
        try:
            tabs = requests.get(f"http://localhost:{self.port}/json", timeout=2).json()
            page_tabs = [t for t in tabs if t.get("type") == "page"]
            self.ws = websocket.create_connection(page_tabs[0]["webSocketDebuggerUrl"], timeout=5)
            return True
        except: return False

    def _send(self, method, params=None):
        if not self.ws: return None
        self._id += 1
        msg = {"id": self._id, "method": method, "params": params or {}}
        try:
            self.ws.send(json.dumps(msg))
            while True:
                resp = json.loads(self.ws.recv())
                if resp.get("id") == self._id: return resp.get("result", {})
        except: self.ws = None; return None

    def read_game_data(self):
        result = self._send("Runtime.evaluate", {"expression": JS_EXTRACT, "returnByValue": True})
        if not result: return {"letterCount": 0, "status": "disconnected"}
        try: return json.loads(result["result"]["value"])
        except: return {"letterCount": 0}

    def focus_input(self, selector):
        self._send("Runtime.evaluate", {"expression": f"document.querySelector('{selector}')?.focus()"})

    def clear_input(self):
        self._send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "a", "windowsVirtualKeyCode": 65, "modifiers": 2})
        self._send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "a", "modifiers": 2})
        self._send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Backspace", "windowsVirtualKeyCode": 8})
        self._send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Backspace"})

    def type_text(self, text):
        import random, time
        for ch in text:
            self._send("Input.dispatchKeyEvent", {"type": "keyDown", "text": ch, "key": ch})
            time.sleep(random.uniform(0.025, 0.05))
            self._send("Input.dispatchKeyEvent", {"type": "keyUp", "key": ch})
            time.sleep(random.uniform(0.02, 0.04))

    def press_enter(self):
        self._send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter", "windowsVirtualKeyCode": 13, "text": "\r"})
        self._send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter"})

    def close(self):
        if self.ws: self.ws.close(); self.ws = None
