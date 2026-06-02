"""
dashboard/app.py — NV-SMART-EDU Teacher Portal
Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import joblib

# ── Import model class (BẮT BUỘC để joblib.load() unpickle đúng object) ───────
# File .pkl được lưu khi PerformanceModel nằm ở module "src.performance_model",
# nên joblib cần import được đúng module này trước khi load, nếu không sẽ báo
# lỗi: ModuleNotFoundError: No module named 'src'
try:
    from src.performance_model import PerformanceModel
    from src.plo_predictor import PLOPredictor
    from src.qa_engine import QAEngine
    from src.data_processor import DataProcessor
    MODEL_IMPORT_ERROR = None
except Exception as e:
    PerformanceModel = None
    PLOPredictor = None
    QAEngine = None
    DataProcessor = None
    MODEL_IMPORT_ERROR = str(e)

from data.vnuk_schema import PLO_DEFINITIONS, PLO_CODES

st.set_page_config(page_title="VN-UK DataScience-EDU", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#f0f4f8;}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#1a9aa6 0%,#0e6b75 60%,#094f58 100%)!important;}
[data-testid="stSidebar"] *{color:#fff!important;}
[data-testid="stSidebar"] .stRadio label{font-size:1rem;padding:0.3rem 0;}
.metric-card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:1.2rem 1.4rem;box-shadow:0 2px 10px rgba(0,0,0,0.06);text-align:center;
  display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:120px;}
.metric-value{font-size:2rem;font-weight:800;color:#0f172a;}
.metric-label{font-size:0.72rem;color:#64748b;text-transform:uppercase;font-weight:600;margin-bottom:0.3rem;}
.metric-danger .metric-value{color:#ef4444;}
.section-title{font-size:1rem;font-weight:700;color:#0f172a;margin:1.4rem 0 0.6rem;
  border-left:4px solid #1a9aa6;padding-left:0.75rem;}
.hero-box{background:linear-gradient(135deg,#0e6b75,#1a9aa6);border-radius:14px;
  padding:2rem 2.5rem;color:white;margin-bottom:1.5rem;}
.login-card{background:white;border-radius:14px;padding:2rem;
  border:1px solid #e2e8f0;box-shadow:0 4px 20px rgba(0,0,0,0.08);max-width:420px;margin:2rem auto;}
.feature-card{background:white;border-radius:15px;padding:1.8rem 1.4rem;border:1px solid #e2e8f0;
  box-shadow:0 2px 8px rgba(0,0,0,.05);min-height:180px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;}
.feature-card .feat-icon{font-size:2.2rem;margin-bottom:0.6rem;}
.feature-card h4{margin:0 0 .5rem;color:#0f172a;font-weight:700;font-size:1rem;}
.feature-card p{margin:0;color:#64748b;font-size:.85rem;line-height:1.4;}
.stButton>button{background:linear-gradient(135deg,#1a9aa6,#0e6b75)!important;
  color:white!important;border:none!important;border-radius:8px!important;
  font-weight:600!important;padding:0.6rem 2rem!important;}
</style>
""", unsafe_allow_html=True)

DB_PATH    = Path(__file__).parent.parent / "data" / "eduquality.db"
MODEL_PATH = Path(__file__).parent.parent / "models" / "performance_model.pkl"
PLO_PREDICTOR_PATH = Path(__file__).parent.parent / "models" / "plo_predictor.pkl"
QA_ENGINE_PATH = Path(__file__).parent.parent / "models" / "qa_engine.pkl"
PROCESSOR_PATH = Path(__file__).parent.parent / "models" / "processor.pkl"

# ── Assessment weights (must match vnuk_schema.py) ────────────────────────────
WEIGHTS = {"cc": 0.10, "bt": 0.20, "gk": 0.30, "ck": 0.40}

GRADE_TABLE = [
    (9.0, 10.0, "A+", 4.0), (8.5, 8.99, "A", 3.7),
    (8.0, 8.49, "B+", 3.5), (7.0, 7.99, "B", 3.0),
    (6.5, 6.99, "C+", 2.5), (5.5, 6.49, "C", 2.0),
    (5.0, 5.49, "D+", 1.5), (4.0, 4.99, "D", 1.0),
    (0.0, 3.99, "F",  0.0),
]

RISK_ORDER = {"Thấp": 0, "Trung bình": 1, "Cao": 2}


def calc_final(cc, bt, gk, ck):
    return round(cc * WEIGHTS["cc"] + bt * WEIGHTS["bt"] +
                 gk * WEIGHTS["gk"] + ck * WEIGHTS["ck"], 2)


def score_to_grade(score):
    for lo, hi, grade, gpa in GRADE_TABLE:
        if lo <= score <= hi:
            return grade, gpa
    return "F", 0.0


def calc_risk(score, vang):
    if score < 4.5 or vang > 6:
        return "Cao"
    if score < 6.0 or vang > 3:
        return "Trung bình"
    return "Thấp"


# ── Model helpers ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained PerformanceModel bundle (RiskClassifier + GPARegressor + DropoutDetector)."""
    if PerformanceModel is None:
        return None
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.session_state["_model_load_error"] = str(e)
        return None


@st.cache_resource
def load_plo_model():
    """Load the trained PLOPredictor model."""
    if PLOPredictor is None:
        return None
    if not PLO_PREDICTOR_PATH.exists():
        return None
    try:
        return joblib.load(PLO_PREDICTOR_PATH)
    except Exception as e:
        st.session_state["_plo_model_load_error"] = str(e)
        return None


@st.cache_resource
def load_qa_engine():
    """Load the trained QAEngine model."""
    if QAEngine is None:
        return None
    if not QA_ENGINE_PATH.exists():
        return None
    try:
        return joblib.load(QA_ENGINE_PATH)
    except Exception as e:
        st.session_state["_qa_engine_load_error"] = str(e)
        return None


@st.cache_resource
def load_processor():
    """Load the trained DataProcessor scaler/imputer."""
    if DataProcessor is None:
        return None
    if not PROCESSOR_PATH.exists():
        return None
    try:
        return joblib.load(PROCESSOR_PATH)
    except Exception as e:
        st.session_state["_processor_load_error"] = str(e)
        return None


# ── DB helpers ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def authenticate(username, password):
    if not DB_PATH.exists():
        return None
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM teachers WHERE teacher_id=? AND password=?",
        (username.strip(), password)
    ).fetchone()
    return dict(row) if row else None


def get_teacher_subjects(teacher_id):
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT ts.ma_mon, sc.ten_mon, sc.so_tin_chi, sc.khoi_kien_thuc
           FROM teacher_subjects ts
           LEFT JOIN subjects_catalog sc ON ts.ma_mon=sc.ma_mon
           WHERE ts.teacher_id=?""",
        conn, params=(teacher_id,)
    )
    return df


def get_students(ma_mon):
    conn = get_conn()
    df = pd.read_sql_query(
        """SELECT ma_sv,hoc_ky,nam,diem_cc,diem_bt,diem_gk,diem_ck,
                  diem_tong,xep_loai,so_lan_vang,dat_mon,risk_level
           FROM students WHERE ma_mon=?""",
        conn, params=(ma_mon,)
    )
    return df


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("logged_in", False), ("teacher", None), ("selected_subject", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

logged_in = st.session_state["logged_in"]
perf_model = load_model()
plo_model = load_plo_model()
qa_engine = load_qa_engine()
processor = load_processor()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding-bottom:1.2rem;border-bottom:1px solid rgba(255,255,255,.2);margin-bottom:1rem;">
      <div style="font-size:2.4rem;">🎓</div>
      <h3 style="margin:0;font-size:1.25rem;font-weight:800;letter-spacing:.05em;">VN-UK DataScience-EDU</h3>
      <p style="margin:0;font-size:.78rem;opacity:.75;">Hệ thống Phân tích Giáo dục</p>
    </div>""", unsafe_allow_html=True)

    if not logged_in:
        nav = st.radio("Navigation", ["🏠 Giới thiệu", "🔑 Đăng nhập"], label_visibility="collapsed")
    else:
        nav = st.radio("Navigation", ["🏠 Giới thiệu", "📚 Môn học", "🤖 AI Assistant"], label_visibility="collapsed")
        st.divider()
        t = st.session_state["teacher"]
        st.markdown(f"**👤 {t['ten_gv']}**")
        st.markdown(f"<small style='opacity:.8;'>{t['teacher_id']} · {t['email']}</small>", unsafe_allow_html=True)
        st.divider()
        status_md = ""
        status_md += "🟢 Model Performance: đã tải<br>" if perf_model is not None else "🔴 Model Performance: lỗi/chưa tải<br>"
        status_md += "🟢 Model PLO Predictor: đã tải<br>" if plo_model is not None else "🔴 Model PLO Predictor: lỗi/chưa tải<br>"
        status_md += "🟢 Model QA Engine: đã tải<br>" if qa_engine is not None else "🔴 Model QA Engine: lỗi/chưa tải<br>"
        st.markdown(f"<small style='opacity:.85;'>{status_md}</small>", unsafe_allow_html=True)
        if st.button("🚪 Đăng xuất"):
            for k in ["logged_in", "teacher", "selected_subject"]:
                st.session_state[k] = False if k == "logged_in" else None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Giới thiệu
# ══════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Giới thiệu":
    st.markdown("""
    <div class="hero-box">
      <h1 style="margin:0;font-size:2rem;font-weight:800;">🎓 VN-UK DataScience-EDU</h1>
      <p style="margin:.4rem 0 0;opacity:.9;font-size:1rem;">Hệ thống Phân tích Chất lượng Giáo dục · Dành cho Giảng viên</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    feats = [
        ("📊", "Dashboard Thống kê", "Theo dõi điểm số, tỷ lệ đạt và nguy cơ rớt môn theo thời gian thực."),
        ("👨‍🎓", "Quản lý Sinh viên", "Danh sách sinh viên kèm điểm thành phần, phân loại rủi ro."),
        ("🤖", "AI Assistant", "Phân tích thông minh và gợi ý cải thiện chất lượng giảng dạy."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], feats):
        col.markdown(f"""
        <div class="feature-card">
          <div class="feat-icon">{icon}</div>
          <h4>{title}</h4>
          <p>{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f0fdfa;border-left:5px solid #1a9aa6;border-radius:8px;
                padding:1.1rem 1.4rem;margin:1.5rem 0;color:#0f4c52;">
      <b>📌 Hướng dẫn sử dụng</b><br><br>
      1. Nhấn <b>Đăng nhập</b> ở menu bên trái và nhập tài khoản giảng viên.<br>
      2. Chọn <b>Môn học</b> bạn phụ trách để xem dashboard chi tiết.<br>
      3. Sử dụng <b>AI Assistant</b> để phân tích và nhận gợi ý cải thiện chất lượng dạy học.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Đăng nhập
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔑 Đăng nhập":
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown("### 🔑 Đăng nhập Hệ thống")
    if not DB_PATH.exists():
        st.error("⚠️ Database chưa được khởi tạo. Chạy: `python data/init_db.py`")
    else:
        st.info("Demo: **GV001** ~ **GV005** | Mật khẩu: **123456**")
        with st.form("login_form"):
            username = st.text_input("Tên đăng nhập", placeholder="VD: GV001")
            password = st.text_input("Mật khẩu", type="password")
            submitted = st.form_submit_button("Đăng nhập →", use_container_width=True)
            if submitted:
                teacher = authenticate(username, password)
                if teacher:
                    st.session_state["logged_in"] = True
                    st.session_state["teacher"] = teacher
                    st.success(f"✅ Chào mừng, {teacher['ten_gv']}!")
                    st.rerun()
                else:
                    st.error("❌ Tên đăng nhập hoặc mật khẩu không đúng.")
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Môn học + Dashboard
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📚 Môn học":
    if not logged_in:
        st.warning("Vui lòng đăng nhập trước."); st.stop()

    teacher = st.session_state["teacher"]
    df_subjects = get_teacher_subjects(teacher["teacher_id"])

    if df_subjects.empty:
        st.warning("Bạn chưa được phân công môn học nào."); st.stop()

    # Subject picker
    st.markdown(f'<p class="section-title">Chọn Môn học phụ trách</p>', unsafe_allow_html=True)
    subject_options = {
        f"{row.ma_mon} — {row.ten_mon or 'N/A'}": row.ma_mon
        for _, row in df_subjects.iterrows()
    }
    chosen_label = st.selectbox("Môn học", list(subject_options.keys()), label_visibility="collapsed")
    selected_ma_mon = subject_options[chosen_label]
    st.session_state["selected_subject"] = selected_ma_mon

    # Load student data
    df = get_students(selected_ma_mon)
    if df.empty:
        st.info("Chưa có dữ liệu sinh viên cho môn này."); st.stop()

    # ── Metrics ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">📊 Tổng quan</p>', unsafe_allow_html=True)
    total = len(df)
    avg_score = df["diem_tong"].mean()
    pass_rate = df["dat_mon"].mean() * 100
    high_risk = (df["risk_level"] == "Cao").sum()

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "👨‍🎓 Tổng sinh viên", f"{total:,}", ""),
        (c2, "📈 Điểm trung bình",  f"{avg_score:.2f}", ""),
        (c3, "✅ Tỷ lệ đạt",        f"{pass_rate:.1f}%", ""),
        (c4, "🔴 Nguy cơ cao",      f"{high_risk} SV",  "metric-danger"),
    ]
    for col, label, value, extra_cls in cards:
        col.markdown(f"""
        <div class="metric-card {extra_cls}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart: Phân bố Điểm tổng (full width) ───────────────────────────────
    st.markdown('<p class="section-title">Phân bố Điểm tổng</p>', unsafe_allow_html=True)
    fig = px.histogram(df, x="diem_tong", nbins=20, color_discrete_sequence=["#1a9aa6"],
                       labels={"diem_tong": "Điểm tổng", "count": "Số SV"})
    fig.update_layout(margin=dict(t=20, b=20), height=300, plot_bgcolor="white",
                      paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True, key=f"hist_diem_tong_{selected_ma_mon}")

    # ── Student table with sort + edit ────────────────────────────────────────
    st.markdown('<p class="section-title">📋 Danh sách Sinh viên & Điểm số</p>', unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        filter_risk = st.selectbox("Lọc theo mức rủi ro", ["Tất cả", "Thấp", "Trung bình", "Cao"])
    with filter_col2:
        sort_dir = st.selectbox("Sắp xếp rủi ro", ["Thấp → Cao", "Cao → Thấp"])

    df_show = df.copy()
    if filter_risk != "Tất cả":
        df_show = df_show[df_show["risk_level"] == filter_risk]

    # Sort by risk level correctly
    df_show["_risk_order"] = df_show["risk_level"].map(RISK_ORDER)
    ascending = sort_dir == "Thấp → Cao"
    df_show = df_show.sort_values("_risk_order", ascending=ascending).drop(columns=["_risk_order"])

    rename_map = {
        "ma_sv": "Mã SV", "hoc_ky": "Học kỳ", "nam": "Năm",
        "diem_cc": "CC", "diem_bt": "BT", "diem_gk": "Giữa kỳ",
        "diem_ck": "Cuối kỳ", "diem_tong": "Điểm tổng",
        "xep_loai": "Xếp loại", "so_lan_vang": "Vắng",
        "dat_mon": "Đạt môn", "risk_level": "Rủi ro",
    }

    # Editable data editor
    df_edit = df_show.rename(columns=rename_map).reset_index(drop=True)
    edited = st.data_editor(
        df_edit,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["Mã SV", "Học kỳ", "Năm", "Điểm tổng", "Xếp loại", "Đạt môn", "Rủi ro"],
        column_config={
            "CC":       st.column_config.NumberColumn(min_value=0, max_value=10, step=0.1, format="%.2f"),
            "BT":       st.column_config.NumberColumn(min_value=0, max_value=10, step=0.1, format="%.2f"),
            "Giữa kỳ": st.column_config.NumberColumn(min_value=0, max_value=10, step=0.1, format="%.2f"),
            "Cuối kỳ": st.column_config.NumberColumn(min_value=0, max_value=10, step=0.1, format="%.2f"),
            "Vắng":    st.column_config.NumberColumn(min_value=0, max_value=30, step=1),
        },
        key=f"editor_{selected_ma_mon}",
    )
    st.caption(f"Hiển thị {len(df_show):,} / {total:,} sinh viên · Có thể chỉnh sửa cột CC, BT, Giữa kỳ, Cuối kỳ, Vắng")

    # Save edits button
    if st.button("💾 Lưu thay đổi", key="save_edits"):
        conn = get_conn()
        reverse_map = {v: k for k, v in rename_map.items()}
        updated_count = 0
        for _, row in edited.iterrows():
            cc = float(row["CC"])
            bt = float(row["BT"])
            gk = float(row["Giữa kỳ"])
            ck = float(row["Cuối kỳ"])
            vang = int(row["Vắng"])
            ma_sv = row["Mã SV"]

            new_total = calc_final(cc, bt, gk, ck)
            grade, gpa = score_to_grade(new_total)
            risk = calc_risk(new_total, vang)
            dat = 1 if new_total >= 4.0 else 0

            conn.execute("""
                UPDATE students
                SET diem_cc=?, diem_bt=?, diem_gk=?, diem_ck=?,
                    diem_tong=?, xep_loai=?, so_lan_vang=?,
                    dat_mon=?, risk_level=?
                WHERE ma_sv=? AND ma_mon=?
            """, (cc, bt, gk, ck, new_total, grade, vang, dat, risk, ma_sv, selected_ma_mon))
            updated_count += 1
        conn.commit()
        st.success(f"✅ Đã cập nhật {updated_count} sinh viên. Điểm tổng, xếp loại và rủi ro được tính lại tự động.")
        st.rerun()

    # ── Add new student ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("➕ Thêm sinh viên mới", expanded=False):
        with st.form("add_student_form", clear_on_submit=True):
            st.markdown("##### Thông tin sinh viên mới")
            ac1, ac2, ac3 = st.columns(3)
            new_masv = ac1.text_input("Mã SV *", placeholder="VD: SV0999")
            new_nam  = ac2.number_input("Năm (1-4)", min_value=1, max_value=4, value=1)
            # Get the semester from existing data for this course
            course_hk = int(df["hoc_ky"].mode().iloc[0]) if not df.empty else 1
            ac3.text_input("Học kỳ (tự động)", value=str(course_hk), disabled=True)

            st.markdown("##### Điểm số")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            new_cc   = sc1.number_input("CC", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
            new_bt   = sc2.number_input("BT", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
            new_gk   = sc3.number_input("Giữa kỳ", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
            new_ck   = sc4.number_input("Cuối kỳ", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
            new_vang = sc5.number_input("Số lần vắng", min_value=0, max_value=30, value=0, step=1)

            add_submitted = st.form_submit_button("✅ Thêm sinh viên", use_container_width=True)
            if add_submitted:
                if not new_masv.strip():
                    st.error("Vui lòng nhập Mã SV.")
                else:
                    conn = get_conn()
                    # Check if student already exists in this course
                    exists = conn.execute(
                        "SELECT 1 FROM students WHERE ma_sv=? AND ma_mon=?",
                        (new_masv.strip(), selected_ma_mon)
                    ).fetchone()
                    if exists:
                        st.error(f"Sinh viên {new_masv} đã tồn tại trong môn {selected_ma_mon}.")
                    else:
                        new_total = calc_final(new_cc, new_bt, new_gk, new_ck)
                        grade, gpa = score_to_grade(new_total)
                        risk = calc_risk(new_total, new_vang)
                        dat = 1 if new_total >= 4.0 else 0

                        # Get course info
                        course_info = conn.execute(
                            "SELECT ten_mon, khoi_kien_thuc, so_tin_chi FROM subjects_catalog WHERE ma_mon=?",
                            (selected_ma_mon,)
                        ).fetchone()
                        ten_mon = course_info["ten_mon"] if course_info else ""
                        khoi = course_info["khoi_kien_thuc"] if course_info else ""
                        tc = course_info["so_tin_chi"] if course_info else 3

                        conn.execute("""
                            INSERT INTO students (
                                ma_sv, nam_hoc, hoc_ky, nam, ma_mon, ten_mon,
                                khoi_kien_thuc, so_tin_chi, profile,
                                diem_cc, diem_bt, diem_gk, diem_ck,
                                diem_tong, xep_loai, diem_gpa,
                                lms_logins, so_lan_vang, ty_le_nop_bai,
                                muc_tham_gia, gio_tu_hoc, ielts_score,
                                dat_mon, dropout_risk, risk_level
                            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (
                            new_masv.strip(), f"2025-2026", course_hk, new_nam,
                            selected_ma_mon, ten_mon, khoi, tc, "manual",
                            round(new_cc, 2), round(new_bt, 2), round(new_gk, 2), round(new_ck, 2),
                            new_total, grade, gpa,
                            0, new_vang, 0.0,
                            0.0, 0.0, 0.0,
                            dat, 0, risk,
                        ))
                        conn.commit()
                        st.success(f"✅ Đã thêm sinh viên **{new_masv}** — Điểm tổng: {new_total:.2f} | Xếp loại: {grade} | Rủi ro: {risk}")
                        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI Assistant
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🤖 AI Assistant":
    if not logged_in:
        st.warning("Vui lòng đăng nhập trước."); st.stop()

    teacher = st.session_state["teacher"]
    selected_ma_mon = st.session_state.get("selected_subject")

    st.markdown("""
    <div class="hero-box">
      <h2 style="margin:0;font-size:1.5rem;font-weight:800;">🤖 AI Assistant</h2>
      <p style="margin:.3rem 0 0;opacity:.85;">Phân tích thông minh hỗ trợ cải thiện chất lượng giảng dạy</p>
    </div>""", unsafe_allow_html=True)

    if not selected_ma_mon:
        st.info("💡 Hãy chọn môn học ở mục **Môn học** trước để nhận phân tích chi tiết.")
        st.stop()

    # Load full student records (including all features and PLO columns)
    conn = get_conn()
    df_all = pd.read_sql_query(
        "SELECT * FROM students WHERE ma_mon=?",
        conn, params=(selected_ma_mon,)
    )
    if df_all.empty:
        st.info("Chưa có dữ liệu để phân tích."); st.stop()

    tab1, tab2, tab3 = st.tabs(["📊 Phân bố Mức rủi ro", "🎯 Phân tích PLO", "💬 Hỏi đáp nhanh"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: Phân bố Mức rủi ro
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<p class="section-title">📊 So sánh Phân bố Mức rủi ro (Database vs AI Model)</p>', unsafe_allow_html=True)
        
        if perf_model is None or processor is None:
            st.warning(
                "⚠️ Model AI (`performance_model.pkl` hoặc `processor.pkl`) chưa được tải thành công. "
                "Đang hiển thị dữ liệu thực tế từ Database."
            )
            # Fallback to pure DB stats
            total = len(df_all)
            high_r = (df_all["risk_level"] == "Cao").sum()
            mid_r = (df_all["risk_level"] == "Trung bình").sum()
            
            ch1, ch2 = st.columns([2, 3])
            with ch1:
                risk_counts = df_all["risk_level"].value_counts().reset_index()
                risk_counts.columns = ["Mức", "Số SV"]
                risk_counts["_order"] = risk_counts["Mức"].map(RISK_ORDER)
                risk_counts = risk_counts.sort_values("_order").drop(columns=["_order"])
                color_map = {"Thấp": "#22c55e", "Trung bình": "#f59e0b", "Cao": "#ef4444"}
                fig2 = px.pie(risk_counts, names="Mức", values="Số SV",
                              color="Mức", color_discrete_map=color_map, hole=0.45,
                              category_orders={"Mức": ["Thấp", "Trung bình", "Cao"]})
                fig2.update_layout(margin=dict(t=20, b=20), height=300, legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig2, use_container_width=True, key=f"pie_risk_fallback_{selected_ma_mon}")
                
            with ch2:
                r1, r2, r3 = st.columns(3)
                r1.markdown(f'<div class="metric-card"><div class="metric-label">🟢 Rủi ro thấp</div><div class="metric-value" style="color:#22c55e;">{total - high_r - mid_r}</div></div>', unsafe_allow_html=True)
                r2.markdown(f'<div class="metric-card"><div class="metric-label">🟡 Rủi ro trung bình</div><div class="metric-value" style="color:#f59e0b;">{mid_r}</div></div>', unsafe_allow_html=True)
                r3.markdown(f'<div class="metric-card"><div class="metric-label">🔴 Rủi ro cao</div><div class="metric-value" style="color:#ef4444;">{high_r}</div></div>', unsafe_allow_html=True)
        else:
            try:
                # Preprocess and run predictions
                X = processor.transform(df_all)
                preds = perf_model.predict(X)
                
                df_all["pred_risk_level"] = preds.get("risk_level", df_all["risk_level"])
                df_all["pred_gpa"] = preds.get("gpa_predicted", [0.0] * len(df_all))
                df_all["pred_dropout_risk"] = preds.get("dropout_risk", [0] * len(df_all))
                df_all["pred_dropout_proba"] = preds.get("dropout_proba", [0.0] * len(df_all))
                df_all["pred_dropout_action"] = preds.get("dropout_action", ["N/A"] * len(df_all))
                
                total = len(df_all)
                high_r_db = (df_all["risk_level"] == "Cao").sum()
                mid_r_db = (df_all["risk_level"] == "Trung bình").sum()
                
                high_r_ai = (df_all["pred_risk_level"] == "Cao").sum()
                mid_r_ai = (df_all["pred_risk_level"] == "Trung bình").sum()
                
                col_db, col_ai = st.columns(2)
                
                with col_db:
                    st.markdown("<h5 style='text-align: center; color:#0e6b75;'>📂 Thực tế (Database)</h5>", unsafe_allow_html=True)
                    risk_counts_db = df_all["risk_level"].value_counts().reset_index()
                    risk_counts_db.columns = ["Mức", "Số SV"]
                    risk_counts_db["_order"] = risk_counts_db["Mức"].map(RISK_ORDER)
                    risk_counts_db = risk_counts_db.sort_values("_order").drop(columns=["_order"])
                    color_map = {"Thấp": "#22c55e", "Trung bình": "#f59e0b", "Cao": "#ef4444"}
                    fig_db = px.pie(risk_counts_db, names="Mức", values="Số SV",
                                    color="Mức", color_discrete_map=color_map, hole=0.45)
                    fig_db.update_layout(margin=dict(t=20, b=20, l=10, r=10), height=250, legend=dict(orientation="h", y=-0.1))
                    st.plotly_chart(fig_db, use_container_width=True, key=f"pie_risk_db_{selected_ma_mon}")
                    
                    r1, r2, r3 = st.columns(3)
                    r1.markdown(f'<div style="text-align:center;padding:0.4rem;background:#f0fdf4;border-radius:6px;"><small style="color:#16a34a;">🟢 THẤP</small><br><strong>{total - high_r_db - mid_r_db}</strong></div>', unsafe_allow_html=True)
                    r2.markdown(f'<div style="text-align:center;padding:0.4rem;background:#fffbeb;border-radius:6px;"><small style="color:#d97706;">🟡 TRUNG BÌNH</small><br><strong>{mid_r_db}</strong></div>', unsafe_allow_html=True)
                    r3.markdown(f'<div style="text-align:center;padding:0.4rem;background:#fef2f2;border-radius:6px;"><small style="color:#dc2626;">🔴 CAO</small><br><strong>{high_r_db}</strong></div>', unsafe_allow_html=True)
                    
                with col_ai:
                    st.markdown("<h5 style='text-align: center; color:#1a9aa6;'>🤖 Dự báo (AI PerformanceModel)</h5>", unsafe_allow_html=True)
                    risk_counts_ai = df_all["pred_risk_level"].value_counts().reset_index()
                    risk_counts_ai.columns = ["Mức", "Số SV"]
                    risk_counts_ai["_order"] = risk_counts_ai["Mức"].map(RISK_ORDER)
                    risk_counts_ai = risk_counts_ai.sort_values("_order").drop(columns=["_order"])
                    fig_ai = px.pie(risk_counts_ai, names="Mức", values="Số SV",
                                    color="Mức", color_discrete_map=color_map, hole=0.45)
                    fig_ai.update_layout(margin=dict(t=20, b=20, l=10, r=10), height=250, legend=dict(orientation="h", y=-0.1))
                    st.plotly_chart(fig_ai, use_container_width=True, key=f"pie_risk_ai_{selected_ma_mon}")
                    
                    a1, a2, a3 = st.columns(3)
                    a1.markdown(f'<div style="text-align:center;padding:0.4rem;background:#f0fdf4;border-radius:6px;"><small style="color:#16a34a;">🟢 THẤP</small><br><strong>{total - high_r_ai - mid_r_ai}</strong></div>', unsafe_allow_html=True)
                    a2.markdown(f'<div style="text-align:center;padding:0.4rem;background:#fffbeb;border-radius:6px;"><small style="color:#d97706;">🟡 TRUNG BÌNH</small><br><strong>{mid_r_ai}</strong></div>', unsafe_allow_html=True)
                    a3.markdown(f'<div style="text-align:center;padding:0.4rem;background:#fef2f2;border-radius:6px;"><small style="color:#dc2626;">🔴 CAO</small><br><strong>{high_r_ai}</strong></div>', unsafe_allow_html=True)
                    
                st.divider()
                
                st.markdown("##### 📋 Bảng Chi tiết Dự báo Hiệu suất & Học lực của Sinh viên")
                df_preds_show = df_all[[
                    "ma_sv", "diem_tong", "risk_level", 
                    "pred_risk_level", "pred_gpa", "pred_dropout_risk", "pred_dropout_proba"
                ]].copy()
                
                df_preds_show["pred_dropout_proba"] = df_preds_show["pred_dropout_proba"].apply(lambda p: f"{p*100:.1f}%")
                df_preds_show["pred_gpa"] = df_preds_show["pred_gpa"].round(2)
                
                df_preds_show.rename(columns={
                    "ma_sv": "Mã SV",
                    "diem_tong": "Điểm tổng",
                    "risk_level": "Rủi ro (DB)",
                    "pred_risk_level": "Rủi ro dự báo (AI)",
                    "pred_gpa": "GPA dự báo (AI)",
                    "pred_dropout_risk": "Nguy cơ bỏ học (AI)",
                    "pred_dropout_proba": "Xác suất bỏ học"
                }, inplace=True)
                
                st.dataframe(df_preds_show, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Lỗi khi chạy dự báo rủi ro: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Phân tích PLO
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<p class="section-title">🎯 Phân tích Chuẩn đầu ra (PLO Achievement Analysis)</p>', unsafe_allow_html=True)
        
        if plo_model is None or processor is None:
            st.warning("⚠️ Model AI (`plo_predictor.pkl` hoặc `processor.pkl`) chưa được tải thành công.")
        else:
            try:
                # Identify evaluated PLOs for current subject
                eval_plos = []
                for p in PLO_CODES:
                    col_name = f"dat_{p}"
                    if col_name in df_all.columns and not df_all[col_name].isna().all():
                        eval_plos.append(p)
                        
                if not eval_plos:
                    st.info("💡 Môn học này không cấu hình đánh giá trực tiếp cho PLO nào trong cơ sở dữ liệu.")
                else:
                    X = processor.transform(df_all)
                    probas = plo_model.predict_proba(X)
                    
                    # Calculate stats
                    stats = []
                    for p in eval_plos:
                        col_name = f"dat_{p}"
                        actual_pass_rate = df_all[col_name].mean() * 100
                        
                        thr = plo_model.thresholds.get(p, 0.5)
                        pred_probs = probas.get(p)
                        if pred_probs is None:
                            pred_pass_rate = 0.0
                        else:
                            pred_pass_rate = np.mean(pred_probs >= thr) * 100
                            
                        stats.append({
                            "PLO": p,
                            "Thực tế (%)": round(actual_pass_rate, 1),
                            "Dự báo (%)": round(pred_pass_rate, 1)
                        })
                        
                    df_stats = pd.DataFrame(stats)
                    
                    # Chart
                    fig_plo = px.bar(df_stats, x="PLO", y=["Thực tế (%)", "Dự báo (%)"],
                                     barmode="group",
                                     color_discrete_sequence=["#1a9aa6", "#ef4444"])
                    fig_plo.update_layout(yaxis_title="Tỷ lệ đạt (%)", plot_bgcolor="white", paper_bgcolor="white",
                                          margin=dict(t=20, b=20))
                    st.plotly_chart(fig_plo, use_container_width=True, key=f"plo_chart_{selected_ma_mon}")
                    
                    # Student level predictions (use enumerate for positional index alignment)
                    st.markdown("##### 📋 Chi tiết Trạng thái đạt PLO của từng Sinh viên (Dự báo AI)")
                    plo_rows = []
                    for pos_idx, (_, row_sv) in enumerate(df_all.iterrows()):
                        sv_id = row_sv["ma_sv"]
                        diem_t = row_sv["diem_tong"]
                        
                        sv_p_info = {"Mã SV": sv_id, "Điểm tổng": diem_t}
                        for p in eval_plos:
                            thr = plo_model.thresholds.get(p, 0.5)
                            prob = float(probas[p][pos_idx]) if p in probas and pos_idx < len(probas[p]) else None
                            if prob is None:
                                p_str = "⚪ Không đánh giá"
                            else:
                                if prob >= 0.75:
                                    status = "🟢 Đạt tốt"
                                elif prob >= thr:
                                    status = "🟡 Đạt"
                                else:
                                    status = "🔴 Nguy cơ"
                                p_str = f"{status} ({prob*100:.0f}%)"
                            sv_p_info[p] = p_str
                        plo_rows.append(sv_p_info)
                        
                    df_sv_plos = pd.DataFrame(plo_rows)
                    st.dataframe(df_sv_plos, use_container_width=True, hide_index=True)
                    
                    # PLO definitions
                    st.divider()
                    st.markdown("##### 📖 Định nghĩa các chuẩn đầu ra (PLOs) được đánh giá:")
                    for p in eval_plos:
                        st.markdown(f"- **{p}**: {PLO_DEFINITIONS.get(p, '')}")
                        
            except Exception as e:
                st.error(f"Lỗi khi chạy phân tích PLO: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: Hỏi đáp nhanh
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<p class="section-title">💬 Hỏi đáp nhanh với Trợ lý AI (QAEngine)</p>', unsafe_allow_html=True)
        
        if qa_engine is None:
            st.warning("⚠️ Model QAEngine chưa được tải. Đang chạy ở chế độ rule-based rút gọn.")
            
            q = st.text_input("Nhập câu hỏi về dữ liệu môn học hiện tại", placeholder="VD: Sinh viên nào có điểm thấp nhất?")
            if q:
                q_lower = q.lower()
                total = len(df_all)
                high_r = (df_all["risk_level"] == "Cao").sum()
                mid_r = (df_all["risk_level"] == "Trung bình").sum()
                
                if "thấp nhất" in q_lower or "thap nhat" in q_lower:
                    row = df_all.loc[df_all["diem_tong"].idxmin()]
                    st.info(f"Sinh viên có điểm thấp nhất: **{row['ma_sv']}** — Điểm tổng: **{row['diem_tong']:.2f}**")
                elif "cao nhất" in q_lower or "cao nhat" in q_lower:
                    row = df_all.loc[df_all["diem_tong"].idxmax()]
                    st.info(f"Sinh viên có điểm cao nhất: **{row['ma_sv']}** — Điểm tổng: **{row['diem_tong']:.2f}**")
                elif "vắng" in q_lower or "vang" in q_lower:
                    row = df_all.loc[df_all["so_lan_vang"].idxmax()]
                    st.info(f"Sinh viên vắng nhiều nhất: **{row['ma_sv']}** — **{row['so_lan_vang']}** buổi vắng")
                elif "nguy cơ" in q_lower or "rủi ro" in q_lower:
                    st.info(f"Có **{high_r}** sinh viên nguy cơ cao và **{mid_r}** nguy cơ trung bình.")
                else:
                    st.warning("Câu hỏi chưa được nhận dạng. Hãy thử: 'thấp nhất', 'cao nhất', 'vắng nhiều', 'nguy cơ'.")
        else:
            try:
                # Load global knowledge base datasets
                @st.cache_data(ttl=300)
                def get_global_qa_data():
                    conn_qa = sqlite3.connect(DB_PATH, check_same_thread=False)
                    df_s = pd.read_sql_query("SELECT * FROM students", conn_qa)
                    # Check if plo_summary table exists
                    table_exists = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='plo_summary'", 
                        conn_qa
                    )
                    if not table_exists.empty:
                        df_p = pd.read_sql_query("SELECT * FROM plo_summary", conn_qa)
                    else:
                        df_p = pd.DataFrame()
                    conn_qa.close()
                    return df_s, df_p
                    
                df_s, df_p = get_global_qa_data()
                
                q_ai = st.text_input("Nhập câu hỏi tiếng Việt về sinh viên hoặc môn học", 
                                     placeholder="VD: Liệt kê sinh viên có GPA dưới 2.5", key="ai_qa_input")
                
                if q_ai:
                    with st.spinner("Đang truy vấn dữ liệu..."):
                        res = qa_engine.predict_single(q_ai, df_s, df_p)
                        
                    st.markdown("#### 🤖 Trợ lý AI trả lời:")
                    st.info(res.get("cau_tra_loi", "Không tìm thấy câu trả lời."))
                    
                    results_table = res.get("ket_qua")
                    if results_table and isinstance(results_table, list) and len(results_table) > 0:
                        st.markdown("##### 📊 Kết quả truy vấn dữ liệu chi tiết:")
                        df_res = pd.DataFrame(results_table)
                        
                        rename_map_qa = {
                            "ma_sv": "Mã SV", "diem_tong": "Điểm tổng", "so_lan_vang": "Vắng",
                            "gpa_tb": "GPA TB", "diem_gpa": "GPA", "diem_tb": "Điểm TB",
                            "ma_mon": "Mã môn", "ten_mon": "Tên môn"
                        }
                        df_res.rename(columns={k: v for k, v in rename_map_qa.items() if k in df_res.columns}, inplace=True)
                        st.dataframe(df_res, use_container_width=True, hide_index=True)
                        
                    with st.expander("🔍 Chi tiết phân tích ngôn ngữ (NLP Engine)", expanded=False):
                        st.write(f"- **Ý định gốc (Intent Classifier):** `{res.get('intent_nb')}`")
                        st.write(f"- **Ý định cuối cùng (Rule Override):** `{res.get('intent_final')}`")
                        st.write(f"- **Độ tin cậy mô hình:** `{res.get('do_tin_cay', 0)*100:.1f}%`")
                        
            except Exception as e:
                st.error(f"Lỗi khi chạy QAEngine: {e}")