"""
data/init_db.py — Khởi tạo SQLite database từ các file CSV
Chạy: python data/init_db.py
"""
import sqlite3
import pandas as pd
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR  = os.path.join(BASE_DIR, "generated")
DB_PATH  = os.path.join(BASE_DIR, "eduquality.db")

TEACHERS = [
    ("GV001", "Nguyen Van An",   "123456", "gv001@vnuk.edu.vn"),
    ("GV002", "Tran Thi Binh",   "123456", "gv002@vnuk.edu.vn"),
    ("GV003", "Le Minh Cuong",   "123456", "gv003@vnuk.edu.vn"),
    ("GV004", "Pham Thi Dung",   "123456", "gv004@vnuk.edu.vn"),
    ("GV005", "Hoang Van Em",    "123456", "gv005@vnuk.edu.vn"),
]

def create_schema(conn):
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            teacher_id TEXT PRIMARY KEY,
            ten_gv     TEXT NOT NULL,
            password   TEXT NOT NULL,
            email      TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS subjects_catalog (
            ma_mon         TEXT PRIMARY KEY,
            ten_mon        TEXT,
            so_tin_chi     INTEGER,
            khoi_kien_thuc TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS teacher_subjects (
            teacher_id TEXT NOT NULL,
            ma_mon     TEXT NOT NULL,
            PRIMARY KEY (teacher_id, ma_mon)
        )
    """)
    conn.commit()
    print("[OK] Schema created.")


def seed_teachers(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM teachers")
    cur.executemany(
        "INSERT OR REPLACE INTO teachers VALUES (?,?,?,?)", TEACHERS
    )
    conn.commit()
    print(f"[OK] Seeded {len(TEACHERS)} teachers.")


def import_csvs(conn):
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not csv_files:
        print(f"[!] No CSV files found in {CSV_DIR}")
        return
    for csv_path in csv_files:
        table_name = os.path.splitext(os.path.basename(csv_path))[0]
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"[OK] {table_name}: {len(df):,} rows")
        except Exception as e:
            print(f"[ERR] {table_name}: {e}")


def build_catalog_and_assign(conn):
    try:
        df = pd.read_sql_query(
            "SELECT DISTINCT ma_mon, ten_mon, so_tin_chi, khoi_kien_thuc FROM students",
            conn,
        )
        df.to_sql("subjects_catalog", conn, if_exists="replace", index=False)
        print(f"[OK] subjects_catalog: {len(df)} subjects.")

        cur = conn.cursor()
        cur.execute("DELETE FROM teacher_subjects")
        subjects = df["ma_mon"].tolist()
        assignments = [
            (TEACHERS[i % len(TEACHERS)][0], ma_mon)
            for i, ma_mon in enumerate(subjects)
        ]
        cur.executemany(
            "INSERT OR REPLACE INTO teacher_subjects VALUES (?,?)", assignments
        )
        conn.commit()
        print(f"[OK] Assigned {len(assignments)} subjects to {len(TEACHERS)} teachers.")
    except Exception as e:
        print(f"[ERR] catalog/assign: {e}")


def create_views(conn):
    cur = conn.cursor()
    cur.execute("DROP VIEW IF EXISTS v_subject_stats")
    cur.execute("""
        CREATE VIEW v_subject_stats AS
        SELECT
            s.ma_mon,
            sc.ten_mon,
            COUNT(s.ma_sv)                                              AS tong_sv,
            ROUND(AVG(s.diem_tong), 2)                                  AS diem_tb,
            ROUND(AVG(s.dat_mon) * 100, 1)                              AS ty_le_dat,
            SUM(CASE WHEN s.risk_level = 'Cao' THEN 1 ELSE 0 END)       AS sv_nguy_co_cao
        FROM students s
        LEFT JOIN subjects_catalog sc ON s.ma_mon = sc.ma_mon
        GROUP BY s.ma_mon
    """)
    cur.execute("DROP VIEW IF EXISTS v_student_detail")
    cur.execute("""
        CREATE VIEW v_student_detail AS
        SELECT s.*, sc.ten_mon
        FROM students s
        LEFT JOIN subjects_catalog sc ON s.ma_mon = sc.ma_mon
    """)
    conn.commit()
    print("[OK] Views created.")


def create_indexes(conn):
    stmts = [
        "CREATE INDEX IF NOT EXISTS idx_students_ma_mon ON students(ma_mon)",
        "CREATE INDEX IF NOT EXISTS idx_students_risk   ON students(risk_level)",
        "CREATE INDEX IF NOT EXISTS idx_ts_teacher      ON teacher_subjects(teacher_id)",
    ]
    for sql in stmts:
        try:
            conn.execute(sql)
        except Exception:
            pass
    conn.commit()
    print("[OK] Indexes created.")


def main():
    print("\n" + "="*50)
    print("  EduQuality-ML — Database Initialization")
    print("="*50 + "\n")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    create_schema(conn)
    seed_teachers(conn)
    import_csvs(conn)
    build_catalog_and_assign(conn)
    create_views(conn)
    create_indexes(conn)
    conn.close()
    print(f"\n[DONE] Database: {DB_PATH}")
    print("\nDemo accounts (all password=123456):")
    for gv in TEACHERS:
        print(f"  {gv[0]} — {gv[1]}")
    print()


if __name__ == "__main__":
    main()
