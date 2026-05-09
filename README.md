# Vocab Guesser

**Vocab Guesser** là một công cụ tự động giải đố từ vựng thời gian thực dành cho các trò chơi từ vựng trên trình duyệt. Công cụ sử dụng AI (Gemini hoặc Local LLM) kết hợp với cơ sở dữ liệu từ vựng khổng lồ để tìm kiếm và nhập đáp án chính xác chỉ trong tích tắc.

---

## Tính năng nổi bật

- **Tốc độ siêu nhanh**: Giải và nhập đáp án chỉ trong ~0.6 giây.
- **Trí tuệ nhân tạo**: Sử dụng mô hình `sentence-transformers` để tìm kiếm từ vựng theo ngữ nghĩa và ngữ cảnh.
- **Quét DOM thời gian thực**: Tự động đọc dữ liệu trò chơi (định dạng ô chữ, gợi ý, nghĩa, ví dụ) qua Chrome DevTools Protocol (CDP).
- **Tự động học tập**: Tự động lưu lại các từ vựng mới và đáp án đúng vào cơ sở dữ liệu để cải thiện độ chính xác cho lần sau.
- **Chế độ Auto hoàn toàn**: Chỉ cần nhấn một phím để bật/tắt chế độ tự động giải mà không cần can thiệp thủ công.
- **Thống kê phiên làm việc**: Theo dõi số câu đã giải, tỉ lệ chính xác và số từ mới đã học được.

---

## Yêu cầu hệ thống

- **Hệ điều hành**: Windows 10/11.
- **Trình duyệt**: Microsoft Edge hoặc Google Chrome.
- **Ngôn ngữ**: Python 3.10 trở lên.

---

## Cài đặt

1. **Clone repository hoặc tải mã nguồn về máy.**
2. **Cài đặt các thư viện cần thiết**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Cấu hình trình duyệt**:
   Công cụ cần trình duyệt chạy ở chế độ Debugging. File `run.bat` đã được cấu hình sẵn để thực hiện việc này cho Microsoft Edge.

---

## Hướng dẫn sử dụng

### Cách 1: Chạy bằng file Batch (Khuyên dùng)
Chạy file `run.bat`. File này sẽ tự động:
1. Tắt các tiến trình Edge đang chạy.
2. Mở Edge với cổng debug `9222`.
3. Khởi chạy script `main.py`.

### Cách 2: Chạy thủ công
1. Mở trình duyệt với tham số: `--remote-debugging-port=9222`
2. Chạy script chính:
   ```bash
   python main.py
   ```

### Điều khiển trong khi chạy:
- **F9**: Bật/Tắt chế độ **AUTO SOLVE**.
- **F10**: Thoát chương trình và xem thống kê.

---

## Cấu trúc dự án

- `main.py`: Luồng xử lý chính và giao diện điều khiển.
- `browser.py`: Giao tiếp với trình duyệt (đọc DOM, gõ phím).
- `solver.py`: Logic xử lý và tìm kiếm từ vựng.
- `config.json`: File cấu hình các thông số (cổng debug, phím tắt, selector...).
- `log.txt`: Nhật ký hoạt động của tool.
- `data.parquet`: Cơ sở dữ liệu từ vựng cục bộ.

---

## Cấu hình (config.json)

Bạn có thể tùy chỉnh các thông số trong file `config.json`:
- `cdp_port`: Cổng debug của trình duyệt (mặc định 9222).
- `hotkey_solve`: Phím tắt bật/tắt auto (mặc định F9).
- `input_selector`: Selector CSS của ô nhập liệu trong game.
- `model`: Tên mô hình AI sử dụng.

---

## Lưu ý
- Công cụ này được phát triển cho mục đích giáo dục và nghiên cứu về tự động hóa trình duyệt và AI.
- Vui lòng sử dụng có trách nhiệm.

