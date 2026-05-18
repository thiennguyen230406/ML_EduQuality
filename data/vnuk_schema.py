"""
vnuk_schema.py
VNUK Data Science 2020 Curriculum Schema
Mã ngành: 7480204DT — 150 tín chỉ — 10 PLOs

Hệ thống đánh giá theo mô hình OBE (Outcome-Based Education)
Tham khảo: BC Hệ Thống Đánh Giá VNUK 2025
"""

# ─── 10 PROGRAM LEARNING OUTCOMES (PLOs) ──────────────────────────────────────
PLO_DEFINITIONS = {
    "PLO1": "Hiểu và áp dụng được các nguyên lý cơ bản, phong cách và ngôn ngữ lập trình phổ biến.",
    "PLO2": "Thiết kế, phát triển và kiểm thử phần mềm đáp ứng các yêu cầu của khách hàng.",
    "PLO3": "Ứng dụng được công nghệ về phần mềm, trí tuệ nhân tạo và khoa học dữ liệu trong thực tiễn.",
    "PLO4": "Giải quyết được các vấn đề phức tạp trong bối cảnh kinh tế, chính trị, xã hội và pháp luật.",
    "PLO5": "Thiết kế, cài đặt và vận hành các kiến trúc phần mềm cho các ứng dụng đa nền tảng.",
    "PLO6": "Giao tiếp và trình bày ý tưởng bằng tiếng Anh, đạt trình độ IELTS 6.0 hoặc tương đương.",
    "PLO7": "Làm việc nhóm và hợp tác hiệu quả.",
    "PLO8": "Có tư duy phản biện, sáng tạo và khởi nghiệp.",
    "PLO9": "Tự định hướng, đưa ra kết luận chuyên môn và có thể bảo vệ được quan điểm cá nhân.",
    "PLO10": "Có đạo đức nghề nghiệp, có khả năng điều phối, quản lý công việc.",
}

PLO_CODES = list(PLO_DEFINITIONS.keys())  # ["PLO1", ..., "PLO10"]

PLO_CATEGORIES = {
    "Kiến thức": ["PLO1", "PLO2", "PLO3", "PLO4", "PLO5"],
    "Kỹ năng":   ["PLO6", "PLO7", "PLO8"],
    "Thái độ":   ["PLO9", "PLO10"],
}

# ─── PROGRAM OBJECTIVES (POs) → PLOs MAPPING ──────────────────────────────────
PO_PLO_MAP = {
    "PO1": ["PLO1", "PLO2", "PLO3", "PLO4", "PLO5", "PLO6"],
    "PO2": ["PLO6", "PLO7", "PLO8", "PLO9"],
    "PO3": ["PLO7", "PLO8", "PLO9", "PLO10"],
}

# ─── GRADING SYSTEM (VNUK scale) ─────────────────────────────────────────────
GRADE_SCALE = {
    "A+": {"min": 9.5, "max": 10.0, "gpa": 4.0, "label": "Xuất sắc"},
    "A":  {"min": 8.5, "max": 9.4,  "gpa": 4.0, "label": "Giỏi"},
    "B+": {"min": 8.0, "max": 8.4,  "gpa": 3.5, "label": "Khá Giỏi"},
    "B":  {"min": 7.0, "max": 7.9,  "gpa": 3.0, "label": "Khá"},
    "C+": {"min": 6.5, "max": 6.9,  "gpa": 2.5, "label": "Trung bình Khá"},
    "C":  {"min": 5.5, "max": 6.4,  "gpa": 2.0, "label": "Trung bình"},
    "D+": {"min": 5.0, "max": 5.4,  "gpa": 1.5, "label": "Dưới trung bình"},
    "D":  {"min": 4.0, "max": 4.9,  "gpa": 1.0, "label": "Yếu"},
    "F":  {"min": 0.0, "max": 3.9,  "gpa": 0.0, "label": "Kém - Không đạt"},
}

PASSING_SCORE = 4.0   # Điểm đạt tối thiểu (D)
GPA_GRADUATION = 2.0  # GPA tốt nghiệp tối thiểu

# ─── ASSESSMENT WEIGHT COMPONENTS ────────────────────────────────────────────
ASSESSMENT_WEIGHTS = {
    "chuyen_can":   0.10,   # Attendance (Formative)
    "bai_tap":      0.20,   # Work Assignment (Formative)
    "giua_ky":      0.30,   # Mid-term Exam (Summative)
    "cuoi_ky":      0.40,   # Final Exam (Summative)
}

# ─── CONTRIBUTION LEVELS (Bloom's Taxonomy — theo PDF trang 5-6) ─────────────
# I = Introduce (Biết, Hiểu)    = 1
# R = Reinforce (Vận dụng, PT)  = 2
# M = Mastery (Đánh giá, ST)    = 3
CONTRIBUTION_LEVELS = {"I": 1, "R": 2, "M": 3}
CONTRIBUTION_LABELS = {
    "I": "Introduce (Giới thiệu)",
    "R": "Reinforce (Củng cố)",
    "M": "Mastery (Thành thạo)",
}
CONTRIBUTION_COLORS = {"I": "#3b82f6", "R": "#f59e0b", "M": "#22c55e"}

# PLO contribution level: I=1, R=2, M=3 (backward-compat: H/M/L still supported)
PLO_LEVEL_WEIGHT = {"H": 1.0, "M": 0.6, "L": 0.3, None: 0.0,
                    "I": 0.33, "R": 0.67, "A": 1.0}  # A = Assess (M+A combo)

# ─── ASSESSMENT BENCHMARKS (Ngưỡng đánh giá — theo PDF trang 11) ─────────────
BENCHMARK_DIRECT    = 5.5   # ĐTB HP ≥ 5.5 → coi là Đạt CLO (trực tiếp)
BENCHMARK_INDIRECT  = 3     # Mức ≥ 3/5 → đạt CLO (gián tiếp)
PLO_PASS_RATE       = 0.75  # ≥ 75% SV đạt → PLO thành phần đạt (trực tiếp)
PLO_PASS_INDIRECT   = 3     # ≥ 3/5 → đạt PLO (gián tiếp)
CLO_PASS_SCORE      = 5.5   # Điểm đạt CLO (10-point scale)

# ─── KNOWLEDGE BLOCKS ─────────────────────────────────────────────────────────
KNOWLEDGE_BLOCKS = {
    1: {"name": "Kiến thức chung (đại cương)", "credits_required": 39, "pct": 26.0},
    2: {"name": "Kiến thức cơ sở ngành",        "credits_required": 28, "pct": 18.67},
    3: {"name": "Kiến thức chuyên ngành",        "credits_required": 40, "pct": 26.67},
    4: {"name": "Tự chọn tự do",                 "credits_required": 23, "pct": 12.00},
    5: {"name": "Thực tập và Đồ án tốt nghiệp",  "credits_required": 20, "pct": 15.33},
}

TOTAL_CREDITS = 150

# ─── COURSE CATALOGUE (87 học phần) ──────────────────────────────────────────
# Format: code → {name, credits, block, plos: {PLO_code: level}}
COURSE_CATALOGUE = {
    # ── Block 1: General Knowledge (39 TC) ──
    "BEB23122": {"name": "Triết học Mác-Lênin",          "credits": 3, "block": 1, "plos": {"PLO10": "L"}},
    "BEB32123": {"name": "Kinh tế chính trị Mác-Lênin",  "credits": 2, "block": 1, "plos": {"PLO7": "M"}},
    "BEB32124": {"name": "Chủ nghĩa xã hội khoa học",    "credits": 2, "block": 1, "plos": {"PLO10": "L"}},
    "BEB22034": {"name": "Tư tưởng Hồ Chí Minh",         "credits": 2, "block": 1, "plos": {"PLO10": "M"}},
    "BEB42125": {"name": "Lịch sử Đảng CSVN",             "credits": 2, "block": 1, "plos": {"PLO10": "L"}},
    "BEB12001": {"name": "Kỹ năng học thuật",             "credits": 2, "block": 1, "plos": {"PLO9": "M", "PLO6": "L"}},
    "BEB12002": {"name": "Tin học ứng dụng",              "credits": 2, "block": 1, "plos": {"PLO1": "L"}},
    "BEB14126": {"name": "Tiếng Anh nền tảng 1",          "credits": 4, "block": 1, "plos": {"PLO6": "H"}},
    "BEB14127": {"name": "Tiếng Anh nền tảng 2",          "credits": 4, "block": 1, "plos": {"PLO6": "H"}},
    "BEB14128": {"name": "Tiếng Anh học thuật nâng cao",  "credits": 4, "block": 1, "plos": {"PLO6": "H"}},
    "BEB14130": {"name": "Chuẩn bị kỹ năng IELTS L1",    "credits": 4, "block": 1, "plos": {"PLO6": "H"}},
    "BEB14131": {"name": "Chuẩn bị kỹ năng IELTS L2",    "credits": 4, "block": 1, "plos": {"PLO6": "H"}},
    "BEB14132": {"name": "Chuẩn bị kỹ năng IELTS L3",    "credits": 4, "block": 1, "plos": {"PLO6": "H"}},
    "BEB14133": {"name": "Chuẩn bị kỹ năng IELTS L4",    "credits": 4, "block": 1, "plos": {"PLO6": "H"}},
    "BEB14137": {"name": "Tiếng Anh học thuật 1",         "credits": 4, "block": 1, "plos": {"PLO6": "H"}},
    "BEB15138": {"name": "Tiếng Anh học thuật 2",         "credits": 5, "block": 1, "plos": {"PLO6": "H"}},
    "BEB15139": {"name": "Tiếng Anh học thuật 3",         "credits": 5, "block": 1, "plos": {"PLO6": "H"}},
    "DSB13002": {"name": "Tiếng Anh trong CNTT",          "credits": 3, "block": 1, "plos": {"PLO6": "H", "PLO1": "L"}},
    "BEB12003": {"name": "Kỹ năng giao tiếp",             "credits": 2, "block": 1, "plos": {"PLO7": "H", "PLO8": "M"}},
    "BEB11003": {"name": "Kỹ năng giao tiếp LT",          "credits": 1, "block": 1, "plos": {"PLO7": "M"}},
    "BEB13004": {"name": "Lập kế hoạch phát triển cá nhân","credits":3, "block": 1, "plos": {"PLO9": "H", "PLO8": "M"}},
    "BEB12010": {"name": "Văn hóa toàn cầu",              "credits": 2, "block": 1, "plos": {"PLO6": "M", "PLO10": "L"}},
    "BEB12013": {"name": "Văn hóa và Giáo dục Anh",       "credits": 2, "block": 1, "plos": {"PLO6": "M"}},
    "BEB13009": {"name": "Khởi sự kinh doanh",            "credits": 3, "block": 1, "plos": {"PLO8": "H", "PLO4": "M"}},
    "BEB13116": {"name": "Tư duy thiết kế",               "credits": 3, "block": 1, "plos": {"PLO8": "H", "PLO2": "M"}},

    # ── Block 2: Core DS Knowledge (28 TC) ──
    "CSB13054": {"name": "Nhập môn lập trình và máy tính","credits": 3, "block": 2, "plos": {"PLO1": "H", "PLO3": "M"}},
    "CSB25017": {"name": "Lập trình hướng đối tượng",     "credits": 5, "block": 2, "plos": {"PLO1": "H", "PLO2": "H"}},
    "CSB25018": {"name": "Phân tích HT & thiết kế CSDL",  "credits": 5, "block": 2, "plos": {"PLO2": "H", "PLO4": "M", "PLO5": "H"}},
    "CSB13055": {"name": "Lập trình Java",                "credits": 3, "block": 2, "plos": {"PLO1": "H", "PLO2": "M"}},
    "DSB23015": {"name": "Thách thức dữ liệu 1",          "credits": 3, "block": 2, "plos": {"PLO3": "H", "PLO7": "M", "PLO8": "M"}},
    "DSB23016": {"name": "Đại số tuyến tính",             "credits": 3, "block": 2, "plos": {"PLO1": "M", "PLO3": "M"}},
    "DSB25030": {"name": "Đại số tuyến tính cho KHDL",    "credits": 5, "block": 2, "plos": {"PLO1": "M", "PLO3": "H"}},
    "DSB13001": {"name": "Xác suất thống kê",             "credits": 3, "block": 2, "plos": {"PLO1": "M", "PLO3": "H", "PLO4": "M"}},
    "CSB22024": {"name": "Thuật toán và cấu trúc dữ liệu","credits": 2, "block": 2, "plos": {"PLO1": "H", "PLO3": "M"}},
    "CSB25020": {"name": "Toán cho khoa học máy tính",    "credits": 5, "block": 2, "plos": {"PLO1": "M", "PLO3": "H"}},

    # ── Block 3: Specialized DS (40 TC) ──
    "DSB23018": {"name": "Phân tích dữ liệu lớn",         "credits": 3, "block": 3, "plos": {"PLO3": "H", "PLO4": "H", "PLO5": "M"}},
    "CSB35037": {"name": "Nhập môn trí tuệ nhân tạo",     "credits": 5, "block": 3, "plos": {"PLO3": "H", "PLO1": "H", "PLO2": "M"}},
    "DSB33020": {"name": "Dự đoán ứng dụng",              "credits": 3, "block": 3, "plos": {"PLO3": "H", "PLO4": "H"}},
    "DSB33021": {"name": "Kỹ thuật mô hình hóa",          "credits": 3, "block": 3, "plos": {"PLO3": "H", "PLO2": "M"}},
    "DSB45027": {"name": "Internet kết nối vạn vật",      "credits": 5, "block": 3, "plos": {"PLO3": "H", "PLO5": "H"}},
    "CSB25022": {"name": "Thiết kế & phát triển ứng dụng web","credits":5,"block": 3,"plos": {"PLO2": "H", "PLO5": "H", "PLO1": "M"}},
    "CSB35036": {"name": "Thiết kế & phát triển ứng dụng di động","credits":5,"block":3,"plos":{"PLO2":"H","PLO5":"H"}},
    "DSB43024": {"name": "Đạo đức trong KHDL",            "credits": 3, "block": 3, "plos": {"PLO10": "H", "PLO4": "M"}},
    "DSB23017": {"name": "Thách thức dữ liệu 2",          "credits": 3, "block": 3, "plos": {"PLO3": "H", "PLO7": "H", "PLO8": "H"}},
    "DSB33019": {"name": "Thách thức dữ liệu 3",          "credits": 3, "block": 3, "plos": {"PLO3": "H", "PLO7": "H", "PLO9": "M"}},
    "DSB45025": {"name": "Thách thức dữ liệu nâng cao",   "credits": 5, "block": 3, "plos": {"PLO3": "H", "PLO8": "H", "PLO9": "H"}},
    "DSB25031": {"name": "Học máy và Học sâu",            "credits": 5, "block": 3, "plos": {"PLO3": "H", "PLO1": "H", "PLO2": "M"}},
    "DSB45028": {"name": "Phân tích dữ liệu cho KHSS",    "credits": 5, "block": 3, "plos": {"PLO3": "H", "PLO4": "H"}},
    "DSB45035": {"name": "Ngôn ngữ tự nhiên",             "credits": 5, "block": 3, "plos": {"PLO3": "H", "PLO1": "H"}},
    "DSB15029": {"name": "Thách thức dữ liệu: khai phá",  "credits": 5, "block": 3, "plos": {"PLO3": "H", "PLO8": "M"}},
    "DSB35032": {"name": "Thách thức dữ liệu: Hệ đề nghị","credits": 5, "block": 3, "plos": {"PLO3": "H", "PLO2": "M"}},
    "DBB33003": {"name": "Ra quyết định dựa trên dữ liệu","credits": 3, "block": 3, "plos": {"PLO3": "H", "PLO4": "H", "PLO9": "M"}},
    "CSB45051": {"name": "Thực tiễn tốt nhất phát triển PM","credits":5,"block": 3, "plos": {"PLO2": "H", "PLO10": "M"}},
    "CSB35040": {"name": "Kiểm thử phần mềm",             "credits": 5, "block": 3, "plos": {"PLO2": "H", "PLO5": "M"}},
    "CSB35038": {"name": "Quản lý dự án phần mềm",        "credits": 5, "block": 3, "plos": {"PLO10": "H", "PLO7": "H"}},

    # ── Block 5: Internship & Graduation ──
    "DSB310022": {"name": "Thực tập nghề nghiệp",         "credits": 10, "block": 5, "plos": {
        "PLO1": "H", "PLO2": "H", "PLO3": "H", "PLO4": "M",
        "PLO5": "M", "PLO6": "M", "PLO7": "H", "PLO8": "H", "PLO9": "H", "PLO10": "H"
    }},
    "DSB410026": {"name": "Đồ án tốt nghiệp",             "credits": 10, "block": 5, "plos": {
        "PLO1": "H", "PLO2": "H", "PLO3": "H", "PLO4": "H",
        "PLO5": "H", "PLO6": "M", "PLO7": "H", "PLO8": "H", "PLO9": "H", "PLO10": "H"
    }},
}

# ─── CORE DS COURSES (used for ML feature selection) ─────────────────────────
CORE_DS_COURSES = [
    "DSB23015", "DSB23018", "CSB35037", "DSB33020", "DSB33021",
    "DSB25031", "DSB45025", "DSB23017", "DSB33019", "DBB33003",
    "DSB45028", "DSB45035",
]

# ─── CLO CATALOGUE (Chuẩn đầu ra Học phần) ──────────────────────────────────
# Format: course_code → list of {id, desc, plo, level, bloom_verb}
CLO_CATALOGUE = {
    "BEB23122": [  # Triết học Mác-Lênin
        {"id": "CLO1", "desc": "Phân tích các khái niệm, phạm trù triết học qua các thời kỳ.",    "plo": "PLO10", "level": "R", "bloom": "Phân tích"},
        {"id": "CLO2", "desc": "Giải thích bản chất thế giới dựa trên quan điểm triết học Mác-Lênin.", "plo": "PLO10", "level": "R", "bloom": "Giải thích"},
        {"id": "CLO3", "desc": "Vận dụng tư duy biện chứng để giải quyết vấn đề thực tiễn.",        "plo": "PLO9",  "level": "I", "bloom": "Vận dụng"},
        {"id": "CLO4", "desc": "Đánh giá vấn đề chính trị-xã hội qua học thuyết kinh tế-xã hội.",  "plo": "PLO10", "level": "R", "bloom": "Đánh giá"},
        {"id": "CLO5", "desc": "Ứng dụng triết học chính trị vào nghiên cứu đạo đức, nhân sinh.",    "plo": "PLO10", "level": "R", "bloom": "Ứng dụng"},
        {"id": "CLO6", "desc": "Phát triển tư duy phản biện và lập luận logic trong nghiên cứu.",    "plo": "PLO9",  "level": "I", "bloom": "Sáng tạo"},
    ],
    "CSB25017": [  # Lập trình hướng đối tượng
        {"id": "CLO1", "desc": "Hiểu và giải thích các khái niệm OOP cơ bản (class, object, method).", "plo": "PLO1", "level": "R", "bloom": "Giải thích"},
        {"id": "CLO2", "desc": "Thiết kế và cài đặt chương trình theo mô hình OOP.",                   "plo": "PLO2", "level": "R", "bloom": "Thiết kế"},
        {"id": "CLO3", "desc": "Ứng dụng kế thừa, đa hình, đóng gói vào bài toán thực tế.",            "plo": "PLO1", "level": "M", "bloom": "Ứng dụng"},
        {"id": "CLO4", "desc": "Phân tích và debug lỗi trong chương trình hướng đối tượng.",           "plo": "PLO2", "level": "M", "bloom": "Phân tích"},
        {"id": "CLO5", "desc": "Sử dụng design patterns cơ bản trong phát triển phần mềm.",            "plo": "PLO2", "level": "M", "bloom": "Vận dụng"},
    ],
    "DSB25031": [  # Học máy và Học sâu
        {"id": "CLO1", "desc": "Hiểu và giải thích các thuật toán học máy cơ bản.",                  "plo": "PLO1", "level": "R", "bloom": "Giải thích"},
        {"id": "CLO2", "desc": "Xây dựng mô hình dự đoán và phân loại bằng scikit-learn.",            "plo": "PLO3", "level": "R", "bloom": "Xây dựng"},
        {"id": "CLO3", "desc": "Thiết kế và huấn luyện mạng nơ-ron nhân tạo (ANN, CNN, RNN).",       "plo": "PLO3", "level": "M", "bloom": "Thiết kế"},
        {"id": "CLO4", "desc": "Đánh giá và cải thiện hiệu suất mô hình (overfitting, regularization).","plo": "PLO3", "level": "M", "bloom": "Đánh giá"},
        {"id": "CLO5", "desc": "Ứng dụng deep learning vào bài toán thực tế (NLP, Computer Vision).",  "plo": "PLO2", "level": "M", "bloom": "Ứng dụng"},
    ],
    "DSB23018": [  # Phân tích dữ liệu lớn
        {"id": "CLO1", "desc": "Mô tả và phân biệt các đặc điểm của dữ liệu lớn (Volume, Velocity, Variety).", "plo": "PLO3", "level": "I", "bloom": "Mô tả"},
        {"id": "CLO2", "desc": "Sử dụng công cụ Hadoop/Spark để xử lý dữ liệu phân tán.",               "plo": "PLO3", "level": "R", "bloom": "Sử dụng"},
        {"id": "CLO3", "desc": "Phân tích và trực quan hóa dữ liệu lớn bằng Python.",                   "plo": "PLO4", "level": "R", "bloom": "Phân tích"},
        {"id": "CLO4", "desc": "Thiết kế pipeline xử lý dữ liệu cho bài toán thực tế.",                 "plo": "PLO5", "level": "M", "bloom": "Thiết kế"},
    ],
    "CSB35037": [  # Nhập môn trí tuệ nhân tạo
        {"id": "CLO1", "desc": "Hiểu lịch sử, xu hướng và ứng dụng của trí tuệ nhân tạo.",             "plo": "PLO1", "level": "I", "bloom": "Nhận biết"},
        {"id": "CLO2", "desc": "Áp dụng thuật toán tìm kiếm (BFS, DFS, A*) để giải bài toán.",         "plo": "PLO3", "level": "R", "bloom": "Áp dụng"},
        {"id": "CLO3", "desc": "Xây dựng hệ thống suy diễn logic và biểu diễn tri thức.",              "plo": "PLO3", "level": "R", "bloom": "Xây dựng"},
        {"id": "CLO4", "desc": "Thiết kế và cài đặt mô hình học máy cơ bản.",                          "plo": "PLO1", "level": "M", "bloom": "Thiết kế"},
        {"id": "CLO5", "desc": "Đánh giá hiệu quả các phương pháp AI trong ứng dụng thực tế.",         "plo": "PLO2", "level": "M", "bloom": "Đánh giá"},
    ],
    "DBB33003": [  # Ra quyết định dựa trên dữ liệu
        {"id": "CLO1", "desc": "Phân tích dữ liệu để hỗ trợ ra quyết định kinh doanh.",                "plo": "PLO3", "level": "R", "bloom": "Phân tích"},
        {"id": "CLO2", "desc": "Xây dựng và diễn giải dashboard quản trị dữ liệu.",                   "plo": "PLO4", "level": "R", "bloom": "Xây dựng"},
        {"id": "CLO3", "desc": "Ứng dụng thống kê suy diễn trong phân tích kinh doanh.",               "plo": "PLO4", "level": "M", "bloom": "Ứng dụng"},
        {"id": "CLO4", "desc": "Lập luận và bảo vệ quyết định dựa trên bằng chứng dữ liệu.",          "plo": "PLO9", "level": "M", "bloom": "Đánh giá"},
    ],
}

# ─── EXAM STRUCTURE TYPES (Cấu trúc đề thi — theo PDF trang 33) ──────────────
EXAM_TYPES = {
    "midterm":   {"label": "Kiểm tra giữa kỳ",   "weight": 0.30, "duration_min": 60},
    "final":     {"label": "Kiểm tra cuối kỳ",   "weight": 0.40, "duration_min": 90},
    "quiz":      {"label": "Kiểm tra nhỏ",        "weight": 0.10, "duration_min": 20},
    "assignment":{"label": "Bài tập lớn/Báo cáo","weight": 0.20, "duration_min": None},
}

# ─── SURVEY CATEGORIES (Loại khảo sát — theo PDF trang 20) ───────────────────
SURVEY_CATEGORIES = {
    "sv":  {"label": "Sinh viên",    "target": "Khảo sát SV đang học"},
    "csv": {"label": "Cựu sinh viên", "target": "Khảo sát sau tốt nghiệp"},
    "ntd": {"label": "Nhà tuyển dụng","target": "Khảo sát đơn vị tuyển dụng"},
}

# ─── BLOOM'S TAXONOMY LEVELS ──────────────────────────────────────────────────
BLOOM_LEVELS = {
    1: {"name": "Nhận biết",   "eng": "Remember",   "verbs": ["Nhận biết", "Liệt kê", "Xác định", "Mô tả"]},
    2: {"name": "Hiểu",        "eng": "Understand",  "verbs": ["Giải thích", "Phân biệt", "Tóm tắt", "So sánh"]},
    3: {"name": "Vận dụng",   "eng": "Apply",       "verbs": ["Sử dụng", "Áp dụng", "Tính toán", "Thực hiện"]},
    4: {"name": "Phân tích",   "eng": "Analyze",     "verbs": ["Phân tích", "Phân loại", "So sánh", "Lý giải"]},
    5: {"name": "Đánh giá",    "eng": "Evaluate",    "verbs": ["Đánh giá", "Phê bình", "Lựa chọn", "Bảo vệ"]},
    6: {"name": "Sáng tạo",   "eng": "Create",      "verbs": ["Thiết kế", "Xây dựng", "Tổng hợp", "Đề xuất"]},
}
# Bloom → Contribution Level mapping (I: 1-2, R: 3-4, M: 5-6)
BLOOM_TO_LEVEL = {1: "I", 2: "I", 3: "R", 4: "R", 5: "M", 6: "M"}

# ─── TEACHING & LEARNING STRATEGIES ──────────────────────────────────────────
TEACHING_STRATEGIES = {
    "direct": ["Giải thích cụ thể", "Thuyết giảng", "Thỉnh giảng", "Dạy học trực tuyến"],
    "indirect": ["Câu hỏi gợi mở", "Giải quyết vấn đề", "Học theo tình huống"],
    "experiential": ["Mô hình", "Thực tập", "Đồ án tốt nghiệp", "Nghiên cứu khoa học"],
    "interactive": ["Tranh luận", "Thảo luận", "Học nhóm"],
    "self_directed": ["Bài tập ở nhà"],
}

# ─── ASSESSMENT METHODS ───────────────────────────────────────────────────────
ASSESSMENT_METHODS = {
    "formative": [
        "Đánh giá chuyên cần",
        "Đánh giá bài tập",
        "Đánh giá thuyết trình",
    ],
    "summative": [
        "Kiểm tra viết",
        "Kiểm tra trắc nghiệm",
        "Báo cáo",
        "Đánh giá thuyết trình",
        "Bảo vệ đồ án tốt nghiệp",
        "Đánh giá đồ án tốt nghiệp",
    ],
}

# ─── SEMESTER STRUCTURE ───────────────────────────────────────────────────────
SEMESTERS = {
    1:  {"year": 1, "term": "HK1", "typical_credits": 15},
    2:  {"year": 1, "term": "HK2", "typical_credits": 18},
    3:  {"year": 2, "term": "HK3", "typical_credits": 18},
    4:  {"year": 2, "term": "HK4", "typical_credits": 18},
    5:  {"year": 3, "term": "HK5", "typical_credits": 18},
    6:  {"year": 3, "term": "HK6", "typical_credits": 18},
    7:  {"year": 4, "term": "HK7", "typical_credits": 18},
    8:  {"year": 4, "term": "HK8", "typical_credits": 17},
    9:  {"year": 4, "term": "TT",  "typical_credits": 10},
}

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def score_to_grade(score: float) -> dict:
    """Convert numeric score (0-10) to grade letter and GPA point."""
    for grade, info in GRADE_SCALE.items():
        if info["min"] <= score <= info["max"]:
            return {"grade": grade, "gpa": info["gpa"], "label": info["label"]}
    return {"grade": "F", "gpa": 0.0, "label": "Không đạt"}


def calc_final_score(chuyen_can, bai_tap, giua_ky, cuoi_ky) -> float:
    """Calculate weighted final score from assessment components."""
    return (
        chuyen_can * ASSESSMENT_WEIGHTS["chuyen_can"] +
        bai_tap    * ASSESSMENT_WEIGHTS["bai_tap"] +
        giua_ky    * ASSESSMENT_WEIGHTS["giua_ky"] +
        cuoi_ky    * ASSESSMENT_WEIGHTS["cuoi_ky"]
    )


def get_plo_weight(course_code: str, plo: str) -> float:
    """Get PLO contribution weight for a given course."""
    course = COURSE_CATALOGUE.get(course_code, {})
    level = course.get("plos", {}).get(plo)
    return PLO_LEVEL_WEIGHT.get(level, 0.0)


def get_courses_for_plo(plo: str, min_level: str = "L") -> list:
    """Return list of course codes that contribute to a given PLO."""
    threshold = PLO_LEVEL_WEIGHT[min_level]
    return [
        code for code, info in COURSE_CATALOGUE.items()
        if PLO_LEVEL_WEIGHT.get(info["plos"].get(plo), 0.0) >= threshold
    ]


def get_clo_list(course_code: str) -> list:
    """Return list of CLOs for a course (from CLO_CATALOGUE)."""
    return CLO_CATALOGUE.get(course_code, [])


def calc_plo_achievement(student_clo_scores: dict, course_code: str) -> dict:
    """
    Tính PLO achievement từ điểm CLO của sinh viên.
    student_clo_scores: {"CLO1": 7.5, "CLO2": 5.0, ...}
    Returns: {"PLO1": {"score": 6.8, "achieved": True}, ...}
    """
    clos = CLO_CATALOGUE.get(course_code, [])
    plo_scores = {}
    for clo in clos:
        score = student_clo_scores.get(clo["id"], 0)
        plo = clo["plo"]
        level_w = CONTRIBUTION_LEVELS.get(clo["level"], 1)
        if plo not in plo_scores:
            plo_scores[plo] = {"weighted_sum": 0, "weight_total": 0}
        plo_scores[plo]["weighted_sum"] += score * level_w
        plo_scores[plo]["weight_total"] += level_w

    result = {}
    for plo, data in plo_scores.items():
        avg = data["weighted_sum"] / data["weight_total"] if data["weight_total"] > 0 else 0
        result[plo] = {
            "score": round(avg, 2),
            "achieved": avg >= CLO_PASS_SCORE,
        }
    return result


if __name__ == "__main__":
    print("=== VNUK DS 2020 Schema ===")
    print(f"Total PLOs: {len(PLO_CODES)}")
    print(f"Total courses: {len(COURSE_CATALOGUE)}")
    print(f"CLO courses defined: {len(CLO_CATALOGUE)}")
    print(f"Total credits: {TOTAL_CREDITS}")
    print(f"Benchmark direct: >= {BENCHMARK_DIRECT}")
    print(f"PLO pass rate: >= {PLO_PASS_RATE*100:.0f}%")
    print("\nPLO definitions:")
    for code, desc in PLO_DEFINITIONS.items():
        print(f"  {code}: {desc[:60]}...")
    print("\nCourses for PLO3 (AI/DS):", get_courses_for_plo("PLO3", "H")[:5])
