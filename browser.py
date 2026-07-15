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

        const normalize = (s) => (s || "") 
            .toLowerCase()
            .normalize("NFD")
            .replace(/[\u0300-\u036f]/g, "")
            .replace(/đ/g, "d")
            .replace(/[:：]/g, "")
            .replace(/\s+/g, " ")
            .trim();

        const root = document.body;
        const mainContent = document.querySelector("main") || document.body;

        // =====================================================
        // 🕵️‍♂️ BỘ GIÁM SÁT CHẠY NGẦM (QUAN TRỌNG)
        // =====================================================
        if (!window.__revealed_observer) {
            window.__last_revealed = "";
            window.__revealed_observer = new MutationObserver(() => {
                const ansSpan = document.querySelector("span.text-2xl.font-bold") || 
                                document.querySelector("span.text-green-600.font-bold") ||
                                document.querySelector("span.text-red-600.font-bold");
                if (ansSpan) {
                    const text = ansSpan.textContent.trim().toUpperCase();
                    if (text && !text.includes("*") && text.length >= 2) {
                        window.__last_revealed = text;
                    }
                }
            });
            window.__revealed_observer.observe(document.body, { childList: true, subtree: true, characterData: true });
        }

        // Reset khi ván mới bắt đầu (thấy nhiều dấu *)
        if (root.textContent.includes("* * *") || root.textContent.includes("[*]")) {
            window.__last_revealed = "";
        }

        let translation_vi = "";
        let hint_vi = "";
        let example_en = "";
        let pos_tag = "";

        // =====================================================
        // TEXT INFO
        // =====================================================

        const labels = [
            ...mainContent.querySelectorAll(
                "span.font-semibold"
            )
        ];

        for (const labelNode of labels) {

            const label =
                normalize(labelNode.textContent);

            if (!label)
                continue;

            let valueNode = null;

            const parent =
                labelNode.parentElement;

            if (parent) {

                const candidates = [
                    ...parent.querySelectorAll("span")
                ].filter(el =>
                    el !== labelNode &&
                    !el.classList.contains(
                        "font-semibold"
                    )
                );

                valueNode =
                    candidates.find(el =>
                        !!el.textContent &&
                        el.textContent.trim()
                    ) || null;
            }

            const value = valueNode
                ? valueNode.textContent
                    .replace(/\n/g, " ")
                    .trim()
                : "";

            if (!value)
                continue;

            if (
                !translation_vi &&
                (
                    label.includes("ban dich") ||
                    label.includes("translation")
                )
            ) {
                translation_vi = value;
            }

            else if (
                !hint_vi &&
                (
                    label.includes("giai thich") ||
                    label.includes("nghia")
                )
            ) {
                hint_vi = value;
            }

            else if (
                !example_en &&
                (
                    label.includes("vi du tieng anh") ||
                    label.includes("example")
                )
            ) {
                example_en = value;
            }

            else if (
                !pos_tag &&
                (
                    label.includes("loai tu") ||
                    label.includes("part of speech") ||
                    label.includes("pos")
                )
            ) {
                pos_tag = value;
            }
        }

        // =====================================================
        // BODY TEXT
        // =====================================================

        const bodyText =
            mainContent.textContent || "";

        // =====================================================
        // LETTER BOXES
        // =====================================================

        const letterBoxes = [
            ...root.querySelectorAll("span")
        ].filter(el => {

            const cls =
                el.className || "";

            const t =
                (el.textContent || "").trim();

            return (
                (
                    cls.includes("w-7") ||
                    cls.includes("h-9")
                ) &&
                (
                    t === "*" ||
                    t === "" ||
                    (
                        t.length === 1 &&
                        /[a-zA-Z]/.test(t)
                    )
                )
            );
        });

        let rows = [];

        letterBoxes.forEach(el => {

            const rect =
                el.getBoundingClientRect();

            let found = false;

            for (const row of rows) {

                if (
                    Math.abs(row.y - rect.y) < 10
                ) {
                    row.items.push(el);
                    found = true;
                    break;
                }
            }

            if (!found) {

                rows.push({
                    y: rect.y,
                    items: [el]
                });
            }
        });

        rows.sort((a, b) => a.y - b.y);

        // =====================================================
        // WORD STRUCTURE DETECTION (HANDLES LINE WRAP)
        // =====================================================
        const allLetters = letterBoxes
            .map(box => ({
                box,
                rect: box.getBoundingClientRect()
            }))
            .sort((a, b) => {
                // Sắp xếp theo Y trước, sau đó theo X
                if (Math.abs(a.rect.y - b.rect.y) > 15) return a.rect.y - b.rect.y;
                return a.rect.x - b.rect.x;
            });

        const wordStructure = [];
        const hints = [];
        const revealedChars = {};
        let totalLetters = 0;
        let globalIndex = 0;

        if (allLetters.length > 0) {
            let currentWordLen = 0;
            const avgW = allLetters.reduce((sum, l) => sum + l.rect.width, 0) / allLetters.length || 28;
            const splitGap = avgW * 0.5;
            
            // Tìm biên phải xa nhất mà bất kỳ ô chữ nào từng đạt tới
            const maxRight = Math.max(...allLetters.map(l => l.rect.right));

            for (let i = 0; i < allLetters.length; i++) {
                const item = allLetters[i];
                const t = item.box.textContent.trim();
                
                totalLetters++;
                currentWordLen++;
                
                if (t && t !== "*" && t.length === 1) {
                    hints.push(`${globalIndex + 1}=${t}`);
                    revealedChars[globalIndex + 1] = t.toUpperCase();
                }
                globalIndex++;

                const next = allLetters[i + 1];
                if (!next) {
                    if (currentWordLen > 0) wordStructure.push(currentWordLen);
                } else {
                    const sameLine = Math.abs(next.rect.y - item.rect.y) < 15;
                    let isNewWord = false;

                    if (sameLine) {
                        const gap = next.rect.x - (item.rect.x + item.rect.width);
                        if (gap > splitGap) isNewWord = true;
                    } else {
                        // LOGIC CHO CĂN GIỮA (CENTERED):
                        // Nếu ô cuối cùng của hàng (item) nằm xa biên phải (maxRight) hơn 0.8 lần độ rộng ô,
                        // chứng tỏ hàng đó vẫn còn chỗ nhưng đã chủ động ngắt (do có dấu cách).
                        // Nếu nó nằm sát biên phải, nghĩa là nó bị ép phải xuống dòng -> Word Wrap.
                        if (item.rect.right < (maxRight - avgW * 0.8)) {
                            isNewWord = true;
                        }
                    }

                    if (isNewWord) {
                        if (currentWordLen > 0) wordStructure.push(currentWordLen);
                        currentWordLen = 0;
                    }
                }
            }
        }

        // =====================================================
        // GAME STATUS
        // =====================================================

        let gameStatus = "playing";

        if (
            bodyText.includes("Bạn đã đoán đúng") ||
            bodyText.includes("Chúc mừng")
        ) {
            gameStatus = "player_correct";
        }

        else if (
            bodyText.includes("Đối thủ đã đoán đúng") ||
            bodyText.includes("Rất tiếc")
        ) {
            gameStatus = "enemy_correct";
        }

        else if (
            bodyText.includes("Hết giờ")
        ) {
            gameStatus = "timeout";
        }

        // =====================================================
        // ANSWER DETECTION
        // =====================================================

        let revealedAnswer = window.__last_revealed || "";

        // -----------------------------------------
        // METHOD: Direct Search by totalLetters
        // -----------------------------------------
        if (!revealedAnswer || revealedAnswer.includes("*")) {
            const allNodes = [...document.querySelectorAll("span, div, h1, h2, b, strong")];
            for (const n of allNodes) {
                const text = n.textContent.trim();
                const clean = text.replace(/[^a-zA-Z]/g, "");
                // Nếu độ dài khớp 100%, không có *, và font chữ lớn
                if (clean.length === totalLetters && totalLetters > 0 && !text.includes("*")) {
                    const style = window.getComputedStyle(n);
                    const fontSize = parseFloat(style.fontSize);
                    if (fontSize >= 20) {
                        revealedAnswer = clean.toUpperCase();
                        break;
                    }
                }
            }
        }

        // =====================================================
        // RETURN
        // =====================================================

        return JSON.stringify({

            status: "ok",

            text:
                bodyText.substring(0, 3000),

            letterCount:
                totalLetters,

            wordStructure:
                wordStructure,

            hints:
                hints.join(", "),

            gameStatus:
                gameStatus,

            revealedAnswer:
                revealedAnswer,

            translation_vi:
                translation_vi,

            hint_vi:
                hint_vi,

            example_en:
                example_en,

            pos_tag:
                pos_tag
        });

    }

    catch(err) {

        return JSON.stringify({
            status: "error",
            message: err.message,
            stack: err.stack
        });
    }

})()"""


class Browser:
    def __init__(self, port=9222):
        self.port = port
        self.ws = None
        self._id = 0

    def connect(self):
        try:
            tabs = requests.get(f"http://127.0.0.1:{self.port}/json", timeout=2).json()
            page_tabs = [t for t in tabs if t.get("type") == "page"]
            if not page_tabs:
                print("  [DEBUG] Không tìm thấy tab nào (type='page'). Hãy mở 1 tab game mới.")
                return False
            self.ws = websocket.create_connection(page_tabs[0]["webSocketDebuggerUrl"], timeout=5)
            return True
        except Exception as e:
            print(f"  [DEBUG] Lỗi kết nối CDP: {e}")
            return False

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
        import time
        # Chiến thuật tốc độ: Từ ngắn gõ siêu tốc (0.005), từ dài gõ nhanh (0.015)
        delay = 0.005 if len(text) < 5 else 0.015
        
        for ch in text:
            self._send("Input.dispatchKeyEvent", {"type": "keyDown", "text": ch, "key": ch})
            time.sleep(delay)
            self._send("Input.dispatchKeyEvent", {"type": "keyUp", "key": ch})
            time.sleep(0.005)

    def press_enter(self):
        self._send("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter", "windowsVirtualKeyCode": 13, "text": "\r"})
        self._send("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter"})

    def close(self):
        if self.ws: self.ws.close(); self.ws = None
