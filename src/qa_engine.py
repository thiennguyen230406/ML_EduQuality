"""
qa_engine.py — Natural-Language Data Query Engine (VNUK DS 2020)
Trả lời nhanh các câu hỏi truy vấn dữ liệu sinh viên bằng tiếng Việt, ví dụ:
  - "Liệt kê sinh viên có GPA dưới 2.5"
  - "Những sinh viên nào vắng trên 5 buổi?"
  - "Sinh viên nào có điểm thấp nhất?"
  - "Có bao nhiêu sinh viên nguy cơ cao?"
  - "Thông tin sinh viên SV0001"
  - "Sinh viên nào chưa đạt PLO3?"

Kiến trúc (giống sentiment_analyzer.py):
  1. Intent Classifier   : TF-IDF + MultinomialNB, huấn luyện trên câu hỏi synthetic
  2. Rule-based fallback : regex keyword matching khi ML không chắc chắn
  3. Entity Extractor    : regex trích cột dữ liệu / toán tử so sánh / ngưỡng số / mã SV
  4. Query Executor      : áp filter lên DataFrame thật (students.csv / plo_summary.csv)
"""
import re
import random
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. TỪ ĐIỂN: cột dữ liệu, toán tử so sánh, intent
# ═══════════════════════════════════════════════════════════════════════════

# metric_key -> (tên cột, nguồn dữ liệu, nhãn tiếng Việt)
#   nguồn 'students' = students.csv (cấp độ môn học/học kỳ)
#   nguồn 'plo'      = plo_summary.csv (cấp độ sinh viên, đã tổng hợp)
METRIC_ALIASES = {
    "gpa":            ("gpa_tb",        "plo",      "GPA"),
    "điểm gpa":       ("gpa_tb",        "plo",      "GPA"),
    "điểm trung bình":("diem_tb",       "plo",      "Điểm trung bình"),
    "điểm tổng":      ("diem_tong",     "students", "Điểm tổng kết"),
    "điểm":           ("diem_tong",     "students", "Điểm tổng kết"),
    "chuyên cần":     ("diem_cc",       "students", "Điểm chuyên cần"),
    "bài tập":        ("diem_bt",       "students", "Điểm bài tập"),
    "giữa kỳ":        ("diem_gk",       "students", "Điểm giữa kỳ"),
    "cuối kỳ":        ("diem_ck",       "students", "Điểm cuối kỳ"),
    "vắng":           ("so_lan_vang",   "students", "Số buổi vắng"),
    "lms":            ("lms_logins",    "students", "Số lần truy cập LMS"),
    "tự học":         ("gio_tu_hoc",    "students", "Giờ tự học"),
    "ielts":          ("ielts_score",   "students", "Điểm IELTS"),
    "nộp bài":        ("ty_le_nop_bai", "students", "Tỉ lệ nộp bài"),
    "tham gia":       ("muc_tham_gia",  "students", "Mức độ tham gia"),
}
DEFAULT_METRIC = "điểm tổng"

# từ khóa toán tử -> ký hiệu so sánh
COMPARATOR_PATTERNS = [
    (r"trở lên|từ .* trở lên|>=|lớn hơn hoặc bằng", ">="),
    (r"trở xuống|<=|nhỏ hơn hoặc bằng",              "<="),
    (r"dưới|thấp hơn|nhỏ hơn|kém hơn|ít hơn",         "<"),
    (r"trên|cao hơn|lớn hơn|nhiều hơn|quá",           ">"),
    (r"bằng|=",                                       "=="),
]

NUMBER_RE     = re.compile(r"(\d+[.,]?\d*)")
STUDENT_ID_RE = re.compile(r"\bSV\s?0*(\d+)\b", re.IGNORECASE)
PLO_RE        = re.compile(r"\bPLO\s?0*(\d+)\b", re.IGNORECASE)

RISK_WORDS   = {"cao": "Cao", "trung bình": "Trung bình", "thấp": "Thấp"}

INTENTS = [
    "list_filter",     # liệt kê SV theo ngưỡng (dưới/trên/...)
    "top_min",         # SV có [metric] thấp nhất
    "top_max",         # SV có [metric] cao nhất
    "count_risk",      # đếm SV theo mức rủi ro
    "list_dropout",    # liệt kê SV nguy cơ bỏ học
    "count_dropout",   # đếm SV nguy cơ bỏ học
    "avg_metric",      # trung bình của 1 chỉ số
    "student_info",    # tra cứu 1 sinh viên cụ thể
    "list_fail",       # SV rớt môn / xếp loại F
    "list_plo_fail",   # SV chưa đạt 1 PLO cụ thể
]


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\sàáâãèéêìíòóôõùúýăắặẳẵặđ.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 2. SINH DỮ LIỆU HUẤN LUYỆN SYNTHETIC (câu hỏi mẫu, gán nhãn intent)
# ═══════════════════════════════════════════════════════════════════════════

def generate_training_data(seed: int = 42) -> list:
    """Sinh danh sách (câu_hỏi, intent) để huấn luyện intent classifier.
    Không cần dữ liệu Q&A thật — chỉ cần các mẫu câu đa dạng theo từng intent."""
    rnd = random.Random(seed)
    metrics = list(METRIC_ALIASES.keys())
    data = []

    def add(intent, templates, n_per_template=6):
        for tpl in templates:
            for _ in range(n_per_template):
                m = rnd.choice(metrics)
                thr = rnd.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5, 6, 7, 8, 60, 0.5])
                q = tpl.format(metric=m, thr=thr)
                data.append((q, intent))

    add("list_filter", [
        "liệt kê sinh viên có {metric} dưới {thr}",
        "những sinh viên nào có {metric} thấp hơn {thr}",
        "danh sách sinh viên {metric} dưới {thr}",
        "cho tôi xem sinh viên nào có {metric} nhỏ hơn {thr}",
        "sinh viên có {metric} trên {thr} là những ai",
        "liệt kê sinh viên {metric} cao hơn {thr}",
        "những ai có {metric} vượt quá {thr}",
        "danh sách sinh viên {metric} từ {thr} trở lên",
        "sinh viên nào {metric} dưới mức {thr}",
        "tìm sinh viên có {metric} lớn hơn {thr}",
    ])

    add("top_min", [
        "sinh viên nào có {metric} thấp nhất",
        "ai có {metric} thấp nhất",
        "{metric} thấp nhất là sinh viên nào",
        "cho tôi sinh viên {metric} kém nhất",
    ])

    add("top_max", [
        "sinh viên nào có {metric} cao nhất",
        "ai có {metric} cao nhất",
        "{metric} cao nhất là sinh viên nào",
        "cho tôi sinh viên {metric} tốt nhất",
    ])

    for _ in range(20):
        data.append((rnd.choice([
            "có bao nhiêu sinh viên nguy cơ cao",
            "số lượng sinh viên rủi ro trung bình",
            "thống kê mức độ rủi ro của sinh viên",
            "bao nhiêu sinh viên thuộc nhóm rủi ro thấp",
            "phân bố mức độ rủi ro như thế nào",
            "đếm số sinh viên theo từng mức rủi ro",
        ]), "count_risk"))

    for _ in range(20):
        data.append((rnd.choice([
            "liệt kê sinh viên có nguy cơ bỏ học",
            "những sinh viên nào có khả năng bỏ học",
            "danh sách sinh viên nguy cơ bỏ học cao",
            "ai có nguy cơ nghỉ học",
        ]), "list_dropout"))

    for _ in range(20):
        data.append((rnd.choice([
            "có bao nhiêu sinh viên nguy cơ bỏ học",
            "tổng số sinh viên có khả năng bỏ học là bao nhiêu",
            "số lượng sinh viên nguy cơ nghỉ học",
        ]), "count_dropout"))

    add("avg_metric", [
        "{metric} trung bình của sinh viên là bao nhiêu",
        "trung bình {metric} là bao nhiêu",
        "{metric} trung bình toàn khóa",
        "tính {metric} trung bình",
    ], n_per_template=8)

    for _ in range(25):
        sv = f"SV{rnd.randint(1, 800):04d}"
        data.append((rnd.choice([
            f"thông tin sinh viên {sv}",
            f"cho tôi biết về sinh viên {sv}",
            f"tra cứu {sv}",
            f"{sv} có kết quả học tập như thế nào",
            f"xem hồ sơ của {sv}",
        ]), "student_info"))

    for _ in range(20):
        data.append((rnd.choice([
            "sinh viên nào bị rớt môn",
            "liệt kê sinh viên xếp loại F",
            "danh sách sinh viên không đạt môn",
            "những ai bị điểm F",
        ]), "list_fail"))

    for _ in range(20):
        plo = rnd.randint(1, 10)
        data.append((rnd.choice([
            f"sinh viên nào chưa đạt PLO{plo}",
            f"liệt kê sinh viên không đạt PLO{plo}",
            f"danh sách sinh viên trượt PLO{plo}",
        ]), "list_plo_fail"))

    rnd.shuffle(data)
    return data


# ═══════════════════════════════════════════════════════════════════════════
# 3. QA ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class QAEngine:
    """Hybrid NLQ engine: TF-IDF+NB intent classifier + regex entity extraction
    + query executor trên DataFrame thật (students.csv / plo_summary.csv)."""

    def __init__(self, cv_folds: int = 5, random_state: int = 42, max_results: int = 20):
        self.cv_folds     = cv_folds
        self.random_state = random_state
        self.max_results  = max_results
        self.nb_pipeline  = None
        self.metrics: dict = {}
        self.is_fitted    = False

    # ── Huấn luyện ────────────────────────────────────────────────────────
    def _build_pipeline(self):
        return Pipeline([
            ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1)),
            ("nb", MultinomialNB(alpha=0.3)),
        ])

    def fit(self, training_data: list = None):
        """training_data: list các tuple (câu_hỏi, intent). Nếu None, tự sinh synthetic."""
        print("\n  Training QA Engine (intent classifier) ...")
        training_data = training_data or generate_training_data(self.random_state)
        texts  = [_clean_text(q) for q, _ in training_data]
        labels = [intent for _, intent in training_data]

        self.nb_pipeline = self._build_pipeline()
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        f1s = cross_val_score(self.nb_pipeline, texts, labels, cv=cv,
                              scoring="f1_weighted", n_jobs=-1)
        self.nb_pipeline.fit(texts, labels)
        self.metrics["intent_clf"] = {
            "cv_f1_weighted": round(float(f1s.mean()), 4),
            "cv_f1_std":      round(float(f1s.std()), 4),
            "n_samples":      len(training_data),
            "n_intents":      len(set(labels)),
        }
        print(f"    ✓ Intent Clf  F1={f1s.mean():.3f}±{f1s.std():.3f}  "
              f"({len(training_data)} mẫu, {len(set(labels))} intent)")
        self.is_fitted = True
        return self

    # ── Trích entity bằng regex ──────────────────────────────────────────
    def _extract_metric(self, text: str):
        """Tìm cụm từ khóa cột dữ liệu dài nhất khớp trong câu hỏi."""
        matches = [k for k in METRIC_ALIASES if k in text]
        if not matches:
            return None
        best = max(matches, key=len)
        return METRIC_ALIASES[best]

    def _extract_comparator(self, text: str) -> str:
        for pattern, op in COMPARATOR_PATTERNS:
            if re.search(pattern, text):
                return op
        return "<"  # mặc định

    def _extract_threshold(self, text: str):
        m = NUMBER_RE.search(text)
        if not m:
            return None
        return float(m.group(1).replace(",", "."))

    def _extract_student_id(self, text: str):
        m = STUDENT_ID_RE.search(text)
        if not m:
            return None
        return f"SV{int(m.group(1)):04d}"

    def _extract_plo(self, text: str):
        m = PLO_RE.search(text)
        if not m:
            return None
        return f"PLO{int(m.group(1))}"

    def _extract_risk(self, text: str):
        for kw, label in RISK_WORDS.items():
            if kw in text:
                return label
        return None

    # ── Rule-based override (đảm bảo intent chắc chắn không bị NB đoán sai) ─
    def _rule_override(self, text: str, nb_intent: str) -> str:
        if self._extract_student_id(text) and "thấp nhất" not in text and "cao nhất" not in text:
            return "student_info"
        if self._extract_plo(text):
            return "list_plo_fail"
        if "bỏ học" in text or "nghỉ học" in text:
            return "count_dropout" if any(w in text for w in ["bao nhiêu", "số lượng", "tổng số"]) else "list_dropout"
        if ("rớt" in text or "trượt môn" in text or "xếp loại f" in text
                or "điểm f" in text or "không đạt môn" in text):
            return "list_fail"
        if "thấp nhất" in text or "kém nhất" in text:
            return "top_min"
        if "cao nhất" in text or "tốt nhất" in text:
            return "top_max"
        if "trung bình" in text and any(k in text for k in METRIC_ALIASES) and "sinh viên nào" not in text:
            return "avg_metric"
        if ("nguy cơ cao" in text or "rủi ro" in text or "phân bố" in text or "thống kê" in text) \
                and "bỏ học" not in text:
            return "count_risk"
        return nb_intent

    # ── Query Executor ────────────────────────────────────────────────────
    def _run_query(self, intent: str, text: str,
                    df_students: pd.DataFrame, df_plo: pd.DataFrame = None) -> dict:
        df_plo = df_plo if df_plo is not None else pd.DataFrame()
        metric_info = self._extract_metric(text) or METRIC_ALIASES[DEFAULT_METRIC]
        col, source, label_vn = metric_info
        src_df = df_students if source == "students" else df_plo

        # nếu nguồn plo được yêu cầu nhưng không có sẵn -> fallback students khi có thể
        if source == "plo" and src_df.empty and col in df_students.columns:
            src_df, source = df_students, "students"

        if intent == "list_filter":
            op  = self._extract_comparator(text)
            thr = self._extract_threshold(text)
            if thr is None or src_df.empty or col not in src_df.columns:
                return {"loai": intent, "loi": "Không xác định được ngưỡng hoặc cột dữ liệu."}
            ops = {"<": src_df[col] < thr, ">": src_df[col] > thr, "==": src_df[col] == thr,
                   ">=": src_df[col] >= thr, "<=": src_df[col] <= thr}
            mask = ops[op]
            result = src_df.loc[mask].sort_values(col, ascending=(op in ["<", "<="]))
            cols = ["ma_sv", col] + (["ma_mon", "ten_mon"] if "ma_mon" in result.columns else [])
            cols = [c for c in cols if c in result.columns]
            result = result[cols].head(self.max_results)
            return {
                "loai": intent, "cot": col, "nhan_cot": label_vn, "toan_tu": op, "nguong": thr,
                "so_luong": int(mask.sum()), "ket_qua": result.to_dict("records"),
                "cau_tra_loi": f"Có **{int(mask.sum())}** sinh viên có {label_vn.lower()} "
                                f"{'dưới' if op in ['<','<='] else 'trên' if op in ['>','>='] else 'bằng'} {thr}"
                                f"{' (hiển thị tối đa ' + str(self.max_results) + ')' if mask.sum() > self.max_results else ''}.",
            }

        if intent in ("top_min", "top_max"):
            if src_df.empty or col not in src_df.columns:
                return {"loai": intent, "loi": "Không có dữ liệu cho chỉ số này."}
            idx = src_df[col].idxmin() if intent == "top_min" else src_df[col].idxmax()
            row = src_df.loc[idx]
            return {
                "loai": intent, "cot": col, "nhan_cot": label_vn,
                "ma_sv": row["ma_sv"], "gia_tri": float(row[col]),
                "cau_tra_loi": f"Sinh viên có {label_vn.lower()} "
                                f"{'thấp' if intent=='top_min' else 'cao'} nhất: **{row['ma_sv']}** "
                                f"— {label_vn}: **{row[col]:.2f}**",
            }

        if intent == "avg_metric":
            if src_df.empty or col not in src_df.columns:
                return {"loai": intent, "loi": "Không có dữ liệu cho chỉ số này."}
            avg = float(src_df[col].mean())
            return {"loai": intent, "cot": col, "nhan_cot": label_vn, "gia_tri_tb": round(avg, 3),
                    "cau_tra_loi": f"{label_vn} trung bình: **{avg:.2f}**"}

        if intent == "count_risk":
            src = df_plo if not df_plo.empty else df_students
            risk = self._extract_risk(text)
            counts = src["risk_level"].value_counts().to_dict()
            if risk:
                n = counts.get(risk, 0)
                return {"loai": intent, "muc": risk, "so_luong": int(n),
                        "cau_tra_loi": f"Có **{n}** sinh viên thuộc nhóm rủi ro **{risk}**."}
            return {"loai": intent, "phan_bo": counts,
                    "cau_tra_loi": "Phân bố rủi ro: " +
                                   ", ".join(f"{k}: **{v}**" for k, v in counts.items())}

        if intent in ("list_dropout", "count_dropout"):
            src = df_plo if not df_plo.empty else df_students
            mask = src["dropout_risk"] == 1
            result = src.loc[mask, ["ma_sv"] + (["gpa_tb"] if "gpa_tb" in src.columns else [])]
            result = result.drop_duplicates(subset="ma_sv").head(self.max_results)
            if intent == "count_dropout":
                return {"loai": intent, "so_luong": int(result["ma_sv"].nunique() if not result.empty else 0),
                        "cau_tra_loi": f"Có **{src.loc[mask,'ma_sv'].nunique()}** sinh viên có nguy cơ bỏ học."}
            return {"loai": intent, "so_luong": int(mask.sum()), "ket_qua": result.to_dict("records"),
                    "cau_tra_loi": f"Có **{src.loc[mask,'ma_sv'].nunique()}** sinh viên nguy cơ bỏ học."}

        if intent == "list_fail":
            if "xep_loai" in df_students.columns:
                mask = df_students["xep_loai"] == "F"
            else:
                mask = df_students["dat_mon"] == 0
            result = df_students.loc[mask, ["ma_sv", "ma_mon", "ten_mon", "diem_tong"]].head(self.max_results)
            return {"loai": intent, "so_luong": int(mask.sum()), "ket_qua": result.to_dict("records"),
                    "cau_tra_loi": f"Có **{int(mask.sum())}** lượt sinh viên bị điểm F / rớt môn."}

        if intent == "list_plo_fail":
            plo = self._extract_plo(text) or "PLO1"
            col_dat = f"dat_{plo}"
            src = df_plo if not df_plo.empty else df_students
            if col_dat not in src.columns:
                return {"loai": intent, "loi": f"Không tìm thấy dữ liệu cho {plo}."}
            mask = src[col_dat] == 0
            result = src.loc[mask, ["ma_sv"]].drop_duplicates().head(self.max_results)
            return {"loai": intent, "plo": plo, "so_luong": int(mask.sum()),
                    "ket_qua": result.to_dict("records"),
                    "cau_tra_loi": f"Có **{int(mask.sum())}** sinh viên chưa đạt **{plo}**."}

        if intent == "student_info":
            sv_id = self._extract_student_id(text)
            if not sv_id:
                return {"loai": intent, "loi": "Không tìm thấy mã sinh viên trong câu hỏi."}
            info = {}
            if not df_plo.empty:
                row = df_plo.loc[df_plo["ma_sv"] == sv_id]
                if not row.empty:
                    r = row.iloc[0]
                    info.update({"gpa_tb": float(r.get("gpa_tb", np.nan)),
                                 "diem_tb": float(r.get("diem_tb", np.nan)),
                                 "risk_level": r.get("risk_level"),
                                 "dropout_risk": bool(r.get("dropout_risk", 0))})
            courses = df_students.loc[df_students["ma_sv"] == sv_id]
            if courses.empty and not info:
                return {"loai": intent, "loi": f"Không tìm thấy sinh viên {sv_id}."}
            info["so_mon_da_hoc"] = int(courses["ma_mon"].nunique())
            ans = (f"**{sv_id}** — GPA: **{info.get('gpa_tb', 'N/A')}**, "
                   f"mức rủi ro: **{info.get('risk_level', 'N/A')}**, "
                   f"nguy cơ bỏ học: **{'Có' if info.get('dropout_risk') else 'Không'}**, "
                   f"số môn đã học: **{info['so_mon_da_hoc']}**")
            return {"loai": intent, "ma_sv": sv_id, "thong_tin": info, "cau_tra_loi": ans}

        return {"loai": "unknown",
                "cau_tra_loi": "Câu hỏi chưa được nhận dạng. Hãy thử: 'GPA thấp dưới 2.5', "
                                "'vắng trên 5 buổi', 'sinh viên nào rủi ro cao', "
                                "'thông tin sinh viên SV0001', 'chưa đạt PLO3'..."}

    # ── API chính ─────────────────────────────────────────────────────────
    def predict_single(self, question: str,
                        df_students: pd.DataFrame, df_plo: pd.DataFrame = None) -> dict:
        """Nhận câu hỏi tiếng Việt + DataFrame(s) thật -> trả kết quả truy vấn."""
        clean = _clean_text(question)

        if self.is_fitted:
            nb_intent = self.nb_pipeline.predict([clean])[0]
            proba     = float(self.nb_pipeline.predict_proba([clean]).max())
        else:
            nb_intent, proba = "unknown", 0.0

        final_intent = self._rule_override(clean, nb_intent)
        result = self._run_query(final_intent, clean, df_students, df_plo)
        result["cau_hoi"]      = question
        result["intent_nb"]    = nb_intent
        result["intent_final"] = final_intent
        result["do_tin_cay"]   = round(proba, 3)
        return result

    def get_metrics(self) -> dict:
        return self.metrics

    def save(self, path: str = None):
        path = path or str(MODELS_DIR / "qa_engine.pkl")
        joblib.dump(self, path)
        sz = Path(path).stat().st_size / 1024
        print(f"  ✓ QAEngine saved → {path}  ({sz:.1f} KB)")

    @staticmethod
    def load(path: str = None) -> "QAEngine":
        path = path or str(MODELS_DIR / "qa_engine.pkl")
        return joblib.load(path)


# ═══════════════════════════════════════════════════════════════════════════
# 4. HUẤN LUYỆN & DEMO KHI CHẠY TRỰC TIẾP
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df_students = pd.read_csv(Path(__file__).parent / "students.csv")
    df_plo      = pd.read_csv(Path(__file__).parent / "plo_summary.csv")

    engine = QAEngine().fit()
    engine.save()

    demo_questions = [
        "Liệt kê sinh viên có GPA dưới 2.5",
        "Những sinh viên nào vắng trên 5 buổi",
        "Sinh viên nào có điểm tổng thấp nhất",
        "Có bao nhiêu sinh viên nguy cơ cao",
        "Liệt kê sinh viên có nguy cơ bỏ học",
        "Thông tin sinh viên SV0001",
        "Sinh viên nào chưa đạt PLO3",
        "GPA trung bình là bao nhiêu",
        "Sinh viên nào bị rớt môn",
    ]
    print("\n  ── DEMO ──")
    for q in demo_questions:
        r = engine.predict_single(q, df_students, df_plo)
        print(f"\n  Q: {q}")
        print(f"  → intent: {r['intent_final']}  (conf={r['do_tin_cay']})")
        print(f"  → {r['cau_tra_loi']}")
