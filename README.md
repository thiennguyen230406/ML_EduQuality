# EduQuality-ML

Hệ thống minh hoạ phân tích chất lượng giáo dục theo Outcome-Based Education (OBE) cho chương trình Data Science của VNUK. Dự án kết hợp dữ liệu mô phỏng, các mô hình học máy và dashboard Streamlit dành cho giảng viên.


## Chức năng chính

- Theo dõi kết quả học tập theo môn: điểm thành phần, tỷ lệ đạt và nhóm rủi ro.
- Dự báo mức độ đạt 10 Program Learning Outcomes (PLO).
- Phân loại rủi ro học tập, dự báo GPA và phát hiện nguy cơ bỏ học.
- Trợ lý AI trả lời các câu hỏi tiếng Việt về dữ liệu sinh viên, ví dụ: `Liệt kê sinh viên có GPA dưới 2.5`.
- Dashboard có đăng nhập, phân môn cho từng giảng viên và trực quan hoá tương tác.

## Công nghệ

Python, Pandas, NumPy, scikit-learn, XGBoost, Joblib, Streamlit, Plotly và SQLite.

## Cấu trúc dự án

```text
EduQuality-ML/
│
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── README.md                      # Tài liệu dự án
│
├── dashboard/                     # Mã nguồn Streamlit Dashboard
│   ├── __init__.py                # Khởi tạo package dashboard
│   └── app.py                     # Entry point của giao diện giảng viên
│
├── data/                          # Schema, dữ liệu mô phỏng và SQLite
│   ├── __init__.py                # Khởi tạo package data
│   ├── generate_data.py           # Sinh dữ liệu mô phỏng theo schema OBE
│   ├── init_db.py                 # Tạo database, tài khoản và phân công môn
│   ├── csv_to_sqlite.py           # Nạp nhanh các tệp CSV vào SQLite
│   ├── vnuk_schema.py             # Cấu trúc OBE: PLO, CLO và danh mục môn học
│   ├── eduquality.db              # Cơ sở dữ liệu SQLite của dashboard
│   ├── eduquality.db-shm          # Tệp bộ nhớ chia sẻ SQLite (WAL)
│   ├── eduquality.db-wal          # Nhật ký ghi SQLite (WAL)
│   └── generated/                 # Dữ liệu đầu ra dạng CSV
│       ├── students.csv           # Điểm và chỉ số học tập của sinh viên
│       ├── plo_summary.csv        # Tổng hợp mức đạt PLO theo sinh viên
│       ├── clo_scores.csv         # Điểm CLO theo học phần
│
├── models/                        # Các mô hình đã huấn luyện
│   ├── performance_model.pkl      # Dự báo hiệu suất và rủi ro học tập
│   ├── plo_predictor.pkl          # Dự báo mức độ đạt PLO
│   └── qa_engine.pkl              # Nhận diện ý định cho trợ lý AI
│
└── src/                           # Logic cốt lõi của Machine Learning
    ├── __init__.py                # Khởi tạo package mã nguồn
    ├── data_processor.py          # Tiền xử lý và trích xuất đặc trưng
    ├── plo_predictor.py           # Mô hình dự báo tiến độ PLO
    ├── performance_model.py       # Mô hình phân loại rủi ro học vụ
    └── qa_engine.py               # Trợ lý truy vấn dữ liệu bằng tiếng Việt
```

Thư mục cục bộ như `.vscode/` và `__pycache__/` không được liệt kê vì chỉ phục vụ cấu hình IDE hoặc bộ nhớ đệm khi chạy Python.

## Cài đặt

Yêu cầu Python 3.9 trở lên.

```bash
git clone <repository-url>
cd EduQuality-ML
python -m venv .venv
```

Kích hoạt môi trường ảo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trên macOS/Linux:

```bash
source .venv/bin/activate
```

Cài đặt thư viện:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Các thư viện được cài đặt gồm NumPy, Pandas, scikit-learn, XGBoost và Joblib
cho xử lý dữ liệu/mô hình; Streamlit và Plotly cho dashboard. SQLite được tích
hợp sẵn trong Python nên không cần cài đặt riêng.

## Chạy nhanh dashboard

Kho mã hiện có sẵn dữ liệu, cơ sở dữ liệu và mô hình demo. Chạy:

```bash
streamlit run dashboard/app.py
```

Sau đó mở địa chỉ Streamlit hiển thị trên terminal (thường là `http://localhost:8501`). Đăng nhập bằng một trong các tài khoản `GV001` đến `GV005`; mật khẩu chung là `123456`.

## Khởi tạo lại dữ liệu và cơ sở dữ liệu

Nếu muốn tạo bộ dữ liệu demo mới, thực hiện theo thứ tự sau từ thư mục gốc dự án:

```bash
python data/generate_data.py
python data/init_db.py
streamlit run dashboard/app.py
```

`data/generate_data.py` sinh các tệp CSV mô phỏng. `data/init_db.py` nạp CSV vào `data/eduquality.db`, tạo danh mục môn học, phân công giảng viên và tài khoản demo.

Để tùy chỉnh quy mô dữ liệu, chạy hàm `generate_all` từ Python:

```bash
python -c "from data.generate_data import generate_all; generate_all(n_students=1200, n_feedback=600, n_content=250, n_courses=4)"
```

Các tham số:

| Tham số | Ý nghĩa | Mặc định |
| --- | --- | --- |
| `n_students` | Số sinh viên mô phỏng | 800 |
| `n_feedback` | Số phản hồi mô phỏng | 500 |
| `n_content` | Số bản ghi học liệu | 200 |
| `n_courses` | Số môn trên mỗi sinh viên | 4 |

## Sử dụng dashboard

1. Đăng nhập bằng tài khoản giảng viên.
2. Vào **Môn học**, chọn học phần được phân công để xem điểm, rủi ro, PLO/CLO và các biểu đồ tổng hợp.
3. Vào **AI Assistant** để truy vấn dữ liệu bằng tiếng Việt. Ví dụ:
   - `Có bao nhiêu sinh viên nguy cơ cao?`
   - `Sinh viên nào vắng trên 5 buổi?`
   - `Thông tin sinh viên SV0001`

## Dữ liệu và mô hình đi kèm

- `data/generated/students.csv`, `plo_summary.csv`, `clo_scores.csv`, `content.csv`: bộ dữ liệu mô phỏng đang được dashboard sử dụng.
- `data/eduquality.db`: cơ sở dữ liệu SQLite dùng trực tiếp bởi dashboard.
- `models/performance_model.pkl`, `plo_predictor.pkl`, `qa_engine.pkl`: các mô hình đã huấn luyện sẵn để chạy demo.


