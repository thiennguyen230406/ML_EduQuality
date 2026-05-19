"""
generate_data.py — Synthetic data generator for EduQuality-ML (VNUK DS 2020)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.vnuk_schema import (
    COURSE_CATALOGUE, PLO_CODES, calc_final_score,
    score_to_grade, PLO_LEVEL_WEIGHT, CLO_CATALOGUE,
    CONTRIBUTION_LEVELS, BENCHMARK_DIRECT, CLO_PASS_SCORE,
    SURVEY_CATEGORIES,
)

SEED = 42
rng = np.random.default_rng(SEED)
OUTPUT_DIR = Path(__file__).parent / "generated"
OUTPUT_DIR.mkdir(exist_ok=True)


def _clamp(val, lo=0.0, hi=10.0):
    return float(np.clip(val, lo, hi))


PROFILES = {
    "excellent": {"prob":0.15,"gk":8.5,"ck":8.8,"cc":9.2,"bt":8.8,"lms":55,"vang":0.5},
    "good":      {"prob":0.25,"gk":7.2,"ck":7.5,"cc":8.0,"bt":7.5,"lms":42,"vang":1.5},
    "average":   {"prob":0.30,"gk":6.0,"ck":6.2,"cc":7.0,"bt":6.2,"lms":30,"vang":3.0},
    "below_avg": {"prob":0.20,"gk":4.8,"ck":4.5,"cc":6.0,"bt":5.0,"lms":20,"vang":5.0},
    "at_risk":   {"prob":0.10,"gk":3.5,"ck":3.0,"cc":5.0,"bt":3.5,"lms":10,"vang":8.0},
}
PNAMES = list(PROFILES.keys())
PPROBS = [v["prob"] for v in PROFILES.values()]

POSITIVE = [
    "Giảng viên giảng dạy rất nhiệt tình và dễ hiểu.",
    "Môn học rất thú vị, có nhiều bài tập thực hành bổ ích.",
    "Tài liệu học tập được cung cấp đầy đủ và chất lượng cao.",
    "Giảng viên luôn sẵn sàng hỗ trợ sinh viên ngoài giờ học.",
    "Nội dung môn học rất phù hợp với thực tiễn công việc.",
    "Phương pháp giảng dạy sáng tạo, kích thích tư duy phản biện.",
    "Môn học giúp tôi hiểu sâu hơn về machine learning và AI.",
]
NEUTRAL = [
    "Môn học khá bình thường, nội dung ở mức trung bình.",
    "Tài liệu cần được cập nhật thêm cho phù hợp với xu hướng mới.",
    "Bài tập có độ khó vừa phải, không quá dễ cũng không quá khó.",
    "Nội dung học tương đối ổn, cần cải thiện thêm phần thực hành.",
]
NEGATIVE = [
    "Giảng viên giảng quá nhanh, khó theo kịp bài.",
    "Tài liệu học tập còn thiếu và không được cập nhật thường xuyên.",
    "Môn học quá nặng lý thuyết, thiếu thực hành.",
    "Bài tập quá nhiều so với thời lượng môn học.",
    "Hệ thống LMS hoạt động không ổn định, khó nộp bài đúng hạn.",
    "Tiêu chí đánh giá không rõ ràng, gây khó khăn cho sinh viên.",
]


def generate_students(n_students=800, n_courses=4):
    print(f"  → Generating {n_students} students x {n_courses} courses ...")
    codes = list(COURSE_CATALOGUE.keys())
    rows = []
    for i in range(n_students):
        sv_id = f"SV{i+1:04d}"
        pname = rng.choice(PNAMES, p=PPROBS)
        p = PROFILES[pname]
        year = int(rng.integers(1, 5))
        study_h = _clamp(rng.normal(25 if pname=="excellent" else 15, 5), 0, 60)
        ielts   = _clamp(rng.normal(5.2 if pname=="excellent" else 4.5, 0.8), 0, 9)
        chosen  = rng.choice(codes, size=n_courses, replace=False)

        for ma_mon in chosen:
            course = COURSE_CATALOGUE[ma_mon]
            # Fixed semester per course (deterministic from course code)
            sem = (sum(ord(c) for c in ma_mon) % 2) + 1
            std = 1.2
            cc  = _clamp(rng.normal(p["cc"], std))
            bt  = _clamp(rng.normal(p["bt"], std))
            gk  = _clamp(rng.normal(p["gk"], std))
            ck  = _clamp(rng.normal(p["ck"], std))
            final = calc_final_score(cc, bt, gk, ck)
            ginfo = score_to_grade(final)
            lms  = max(0, int(rng.normal(p["lms"], 8)))
            vang = max(0, int(rng.normal(p["vang"], 1.5)))
            nop  = _clamp(rng.normal(0.85 if final>=6 else 0.55, 0.15), 0, 1)
            tham = _clamp(rng.normal(p["bt"], 1.0))

            plo_row = {}
            for plo in PLO_CODES:
                lv = course["plos"].get(plo)
                w  = PLO_LEVEL_WEIGHT.get(lv, 0.0)
                if w == 0:
                    plo_row[plo] = None
                else:
                    prob = min((final/10.0)*w + 0.1, 0.97)
                    plo_row[plo] = int(rng.random() < prob)

            # Risk
            avg_vang = vang
            dropout = 1 if (final < 4.0 and vang > 5) else 0
            if final < 4.5 or vang > 6:   risk = "Cao"
            elif final < 6.0 or vang > 3: risk = "Trung bình"
            else:                          risk = "Thấp"

            row = {
                "ma_sv": sv_id, "nam_hoc": f"202{year}-202{year+1}",
                "hoc_ky": sem, "nam": year,
                "ma_mon": ma_mon, "ten_mon": course["name"],
                "khoi_kien_thuc": course["block"], "so_tin_chi": course["credits"],
                "profile": pname,
                "diem_cc": round(cc,2), "diem_bt": round(bt,2),
                "diem_gk": round(gk,2), "diem_ck": round(ck,2),
                "diem_tong": round(final,2), "xep_loai": ginfo["grade"],
                "diem_gpa": ginfo["gpa"],
                "lms_logins": lms, "so_lan_vang": vang,
                "ty_le_nop_bai": round(nop,3), "muc_tham_gia": round(tham,2),
                "gio_tu_hoc": round(study_h,1), "ielts_score": round(ielts,1),
                "dat_mon": int(final>=4.0), "dropout_risk": dropout, "risk_level": risk,
                **{f"dat_{plo}": plo_row[plo] for plo in PLO_CODES},
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR/"students.csv", index=False, encoding="utf-8-sig")
    print(f"  ✓ students.csv   → {len(df):,} rows")
    return df


def generate_feedback(n=500):
    print(f"  → Generating {n} feedback records ...")
    codes   = list(COURSE_CATALOGUE.keys())
    sentmap = {"positive": POSITIVE, "neutral": NEUTRAL, "negative": NEGATIVE}
    labmap  = {"positive": 1, "neutral": 0, "negative": -1}
    ratmap  = {"positive": (7,10), "neutral": (5,7), "negative": (1,5)}
    sents   = rng.choice(["positive","neutral","negative"], size=n, p=[0.45,0.30,0.25])
    rows = []
    for i, sent in enumerate(sents):
        phrases = sentmap[sent]
        comment = str(rng.choice(phrases))
        if rng.random() < 0.3:
            comment += " " + str(rng.choice(phrases))
        lo, hi = ratmap[sent]
        rows.append({
            "feedback_id": f"FB{i+1:04d}",
            "ma_mon": str(rng.choice(codes)),
            "hoc_ky": int(rng.integers(1,3)),
            "topic": str(rng.choice(["giang_vien","tai_lieu","phuong_phap",
                                      "co_so_vat_chat","bai_tap","noi_dung"])),
            "noi_dung": comment,
            "rating": round(float(rng.uniform(lo, hi)), 1),
            "sentiment_label": sent,
            "sentiment_score": labmap[sent],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR/"feedback.csv", index=False, encoding="utf-8-sig")
    print(f"  ✓ feedback.csv   → {len(df):,} rows")
    return df


def generate_content(n=200):
    print(f"  → Generating {n} content quality records ...")
    codes = list(COURSE_CATALOGUE.keys())
    rows = []
    for i in range(n):
        code   = str(rng.choice(codes))
        course = COURSE_CATALOGUE[code]
        so_trang = int(rng.integers(5,120))
        so_hinh  = int(rng.integers(0,40))
        so_vi_du = int(rng.integers(0,25))
        so_bt    = int(rng.integers(0,30))
        do_kho   = int(rng.integers(1,6))
        nam_cap  = int(rng.integers(2018,2025))
        tai_lieu = int(rng.random()<0.7)
        rubric   = int(rng.random()<0.6)
        muc_tieu = int(rng.random()<0.75)
        thuc_hanh= round(float(rng.uniform(0,0.6)),2)

        fresh   = max(0,(nam_cap-2018)/6)*2.0
        rich    = min(so_hinh/20+so_vi_du/15+so_bt/20,1.0)*3.0
        cover   = (tai_lieu+rubric+muc_tieu)/3*2.5
        prac    = thuc_hanh*2.5
        score   = _clamp(fresh+rich+cover+prac+rng.normal(0,0.3))
        rows.append({
            "content_id":f"CT{i+1:04d}", "ma_mon":code, "ten_mon":course["name"],
            "so_trang":so_trang, "so_hinh":so_hinh, "so_vi_du":so_vi_du,
            "so_bai_tap":so_bt, "do_kho":do_kho, "cap_nhat_nam":nam_cap,
            "co_tai_lieu_tham":tai_lieu, "co_rubric":rubric, "co_muc_tieu_ro":muc_tieu,
            "ty_le_thuc_hanh":thuc_hanh, "chat_luong_score":round(score,2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR/"content.csv", index=False, encoding="utf-8-sig")
    print(f"  ✓ content.csv    → {len(df):,} rows")
    return df


def generate_plo_summary(students_df):
    print("  → Building PLO summary ...")
    rows = []
    for sv_id, g in students_df.groupby("ma_sv"):
        row = {
            "ma_sv": sv_id,
            "gpa_tb": round(g["diem_gpa"].mean(),2),
            "diem_tb": round(g["diem_tong"].mean(),2),
            "dropout_risk": int(g["dropout_risk"].max()),
            "risk_level": g["risk_level"].iloc[0],
            "profile": g["profile"].iloc[0],
            "nam": int(g["nam"].iloc[0]),
        }
        for plo in PLO_CODES:
            col = f"dat_{plo}"
            vals = g[col].dropna()
            row[f"pct_{plo}"] = round(float(vals.mean()),3) if len(vals)>0 else None
            row[f"dat_{plo}"] = int(vals.mean()>=0.5) if len(vals)>0 else None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR/"plo_summary.csv", index=False, encoding="utf-8-sig")
    print(f"  ✓ plo_summary.csv → {len(df):,} rows")
    return df


def generate_clo_scores(students_df, n_per_course=3):
    """Tạo điểm CLO cho từng sinh viên theo từng học phần có CLO."""
    print(f"  → Generating CLO scores ...")
    clo_courses = list(CLO_CATALOGUE.keys())
    rows = []
    for _, sv_row in students_df.drop_duplicates("ma_sv").iterrows():
        sv_id   = sv_row["ma_sv"]
        base    = sv_row["diem_tong"]  # base performance
        chosen  = rng.choice(clo_courses, size=min(n_per_course, len(clo_courses)), replace=False)
        for code in chosen:
            clos = CLO_CATALOGUE[code]
            for clo in clos:
                # Score influenced by base + noise + level difficulty
                level_penalty = {"I": 0.3, "R": 0.0, "M": -0.5}.get(clo["level"], 0)
                score = float(np.clip(rng.normal(base + level_penalty, 1.2), 0, 10))
                achieved = score >= CLO_PASS_SCORE
                rows.append({
                    "ma_sv":    sv_id,
                    "ma_mon":   code,
                    "ten_mon":  COURSE_CATALOGUE.get(code, {}).get("name", code),
                    "clo_id":   clo["id"],
                    "clo_desc": clo["desc"][:60],
                    "plo":      clo["plo"],
                    "level":    clo["level"],
                    "bloom":    clo["bloom"],
                    "diem_clo": round(score, 2),
                    "dat_clo":  int(achieved),
                    "exam_type": str(rng.choice(["midterm", "final", "assignment"])),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "clo_scores.csv", index=False, encoding="utf-8-sig")
    print(f"  ✓ clo_scores.csv  → {len(df):,} rows")
    return df


def generate_exam_structure():
    """Tạo cấu trúc đề thi theo CLO cho các học phần."""
    print("  → Generating exam structure ...")
    rows = []
    for code, clos in CLO_CATALOGUE.items():
        course_name = COURSE_CATALOGUE.get(code, {}).get("name", code)
        for exam_type in ["midterm", "final"]:
            n_questions = int(rng.integers(20, 50))
            # Distribute questions across CLOs
            weights = rng.dirichlet(np.ones(len(clos)))
            for idx, clo in enumerate(clos):
                n_q = max(1, int(round(weights[idx] * n_questions)))
                pts_per_q = round(10.0 / n_questions, 2)
                rows.append({
                    "ma_mon":         code,
                    "ten_mon":        course_name,
                    "exam_type":      exam_type,
                    "clo_id":         clo["id"],
                    "clo_desc":       clo["desc"][:60],
                    "plo":            clo["plo"],
                    "level":          clo["level"],
                    "bloom":          clo["bloom"],
                    "so_cau":         n_q,
                    "diem_moi_cau":   pts_per_q,
                    "tong_diem_clo":  round(n_q * pts_per_q, 2),
                    "ty_le_clo":      round(weights[idx] * 100, 1),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "exam_structure.csv", index=False, encoding="utf-8-sig")
    print(f"  ✓ exam_structure.csv → {len(df):,} rows")
    return df


def generate_survey_data(n=300):
    """Tạo dữ liệu khảo sát (SV, CSV, NTD) theo mỔ hình PDF trang 20."""
    print(f"  → Generating survey data ({n} records) ...")
    plo_survey_items = [
        {"plo": "PLO1", "question": "Kiến thức lập trình tại trường đáp ứng tốt yêu cầu công việc"},
        {"plo": "PLO2", "question": "Khả năng thiết kế và phát triển phần mềm thực tế"},
        {"plo": "PLO3", "question": "Năng lực ứng dụng AI/KHDL vào giải quyết vấn đề"},
        {"plo": "PLO6", "question": "Kỹ năng giao tiếp tiếng Anh trong môi trường làm việc"},
        {"plo": "PLO7", "question": "Khả năng làm việc nhóm và hợp tác"},
        {"plo": "PLO8", "question": "Tư duy sáng tạo và khả năng khởi nghiệp"},
        {"plo": "PLO9", "question": "Khả năng tự định hướng và học tập liên tục"},
        {"plo": "PLO10", "question": "Đạo đức nghề nghiệp và trach nhiệm xã hội"},
    ]
    category_keys = list(SURVEY_CATEGORIES.keys())
    rows = []
    for i in range(n):
        cat = str(rng.choice(category_keys))
        # Different rating distributions per category
        if cat == "sv":
            base_rating = float(rng.uniform(2.5, 4.5))
        elif cat == "csv":
            base_rating = float(rng.uniform(3.0, 5.0))
        else:  # ntd
            base_rating = float(rng.uniform(2.8, 4.8))

        item = dict(rng.choice(plo_survey_items))
        rating = float(np.clip(rng.normal(base_rating, 0.5), 1, 5))
        rows.append({
            "survey_id":  f"SRV{i+1:04d}",
            "category":   cat,
            "category_label": SURVEY_CATEGORIES[cat]["label"],
            "nam_khao_sat": int(rng.integers(2022, 2026)),
            "hoc_ky":     int(rng.integers(1, 3)),
            "plo":        item["plo"],
            "question":   item["question"],
            "rating":     round(rating, 1),
            "dat":        int(rating >= 3.0),
            "comment":    str(rng.choice([
                "Chương trình đào tạo phù hợp với thực tế.",
                "Cần cập nhật thêm kiến thức công nghệ mới.",
                "Sinh viên có khả năng học hỏi nhanh.",
                "Kỹ năng thực hành cần được tăng cường.",
                "Chất lượng đào tạo tương xứng với khả năng tiếp cận việc làm.",
            ])),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "survey.csv", index=False, encoding="utf-8-sig")
    print(f"  ✓ survey.csv      → {len(df):,} rows")
    return df


def generate_all(n_students=800, n_feedback=500, n_content=200, n_courses=4):
    print("\n" + "="*55)
    print("  EduQuality-ML — Synthetic Data Generator")
    print("  VNUK Data Science 2020 Curriculum")
    print("="*55)
    s = generate_students(n_students, n_courses)
    f = generate_feedback(n_feedback)
    c = generate_content(n_content)
    p = generate_plo_summary(s)
    clo = generate_clo_scores(s)
    exam = generate_exam_structure()
    survey = generate_survey_data(300)
    print(f"\n  ✅ Saved to: {OUTPUT_DIR}\n")
    return s, f, c, p, clo, exam, survey


if __name__ == "__main__":
    generate_all()
