"""
data_processor.py — Feature engineering & preprocessing for EduQuality-ML
Handles KNN imputation, scaling, and VNUK-aligned feature construction.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.vnuk_schema import PLO_CODES, ASSESSMENT_WEIGHTS

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ─── FEATURE SETS ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "diem_cc", "diem_bt", "diem_gk", "diem_ck", "diem_tong",
    "lms_logins", "so_lan_vang", "ty_le_nop_bai",
    "muc_tham_gia", "gio_tu_hoc", "ielts_score",
    "so_tin_chi", "khoi_kien_thuc",
]

CATEGORICAL_FEATURES = ["ma_mon", "hoc_ky", "nam"]

ENGAGEMENT_FEATURES = [
    "lms_logins", "so_lan_vang", "ty_le_nop_bai", "muc_tham_gia",
]

ACADEMIC_FEATURES = [
    "diem_cc", "diem_bt", "diem_gk", "diem_ck", "diem_tong",
    "diem_gpa", "gio_tu_hoc",
]


class DataProcessor:
    """
    VNUK-aware data preprocessor.
    Applies KNN imputation, feature engineering, and scaling.
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.imputer     = KNNImputer(n_neighbors=n_neighbors)
        self.scaler      = StandardScaler()
        self.label_encoders: dict = {}
        self.feature_names: list = []
        self.is_fitted = False

    # ── Feature Engineering ───────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Engagement index (0-1)
        df["engagement_index"] = (
            df["lms_logins"].clip(0,100) / 100 * 0.4 +
            (1 - df["so_lan_vang"].clip(0,15) / 15) * 0.3 +
            df["ty_le_nop_bai"].clip(0,1) * 0.3
        )

        # Academic momentum: weighted recent vs early scores
        df["academic_gap"] = df["diem_ck"] - df["diem_gk"]

        # Weighted score ratio
        df["formative_score"] = (
            df["diem_cc"] * ASSESSMENT_WEIGHTS["chuyen_can"] +
            df["diem_bt"] * ASSESSMENT_WEIGHTS["bai_tap"]
        ) / (ASSESSMENT_WEIGHTS["chuyen_can"] + ASSESSMENT_WEIGHTS["bai_tap"])

        df["summative_score"] = (
            df["diem_gk"] * ASSESSMENT_WEIGHTS["giua_ky"] +
            df["diem_ck"] * ASSESSMENT_WEIGHTS["cuoi_ky"]
        ) / (ASSESSMENT_WEIGHTS["giua_ky"] + ASSESSMENT_WEIGHTS["cuoi_ky"])

        # Study efficiency
        df["study_efficiency"] = df["diem_tong"] / (df["gio_tu_hoc"].clip(1, 60))

        # Attendance rate (assuming 15 sessions/semester)
        df["attendance_rate"] = 1.0 - (df["so_lan_vang"].clip(0, 15) / 15.0)

        # LMS activity normalised
        df["lms_norm"] = df["lms_logins"].clip(0, 80) / 80.0

        # Risk indicators
        df["fail_risk"]    = (df["diem_tong"] < 5.0).astype(int)
        df["absent_heavy"] = (df["so_lan_vang"] > 5).astype(int)
        df["low_submit"]   = (df["ty_le_nop_bai"] < 0.6).astype(int)

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is not None:
                    vals = df[col].astype(str)
                    known = set(le.classes_)
                    vals = vals.apply(lambda x: x if x in known else le.classes_[0])
                    df[col + "_enc"] = le.transform(vals)
        return df

    def _select_features(self, df: pd.DataFrame) -> list:
        engineered = [
            "engagement_index", "academic_gap", "formative_score",
            "summative_score", "study_efficiency", "attendance_rate",
            "lms_norm", "fail_risk", "absent_heavy", "low_submit",
        ]
        categorical_enc = [c + "_enc" for c in CATEGORICAL_FEATURES if c + "_enc" in df.columns]
        base = [c for c in NUMERIC_FEATURES if c in df.columns]
        return base + engineered + categorical_enc

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=True)
        self.feature_names = self._select_features(df)
        X = df[self.feature_names].values.astype(float)
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("DataProcessor not fitted. Call fit_transform() first.")
        df = self._engineer_features(df)
        df = self._encode_categoricals(df, fit=False)
        missing = [f for f in self.feature_names if f not in df.columns]
        for m in missing:
            df[m] = 0.0
        X = df[self.feature_names].values.astype(float)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return X

    def save(self, path: str = None):
        path = path or str(MODELS_DIR / "processor.pkl")
        joblib.dump(self, path)
        print(f"  ✓ Processor saved → {path}")

    @staticmethod
    def load(path: str = None) -> "DataProcessor":
        path = path or str(MODELS_DIR / "processor.pkl")
        return joblib.load(path)

    def get_feature_names(self) -> list:
        return self.feature_names


# ─── HELPER: prepare risk labels ──────────────────────────────────────────────

def encode_risk_labels(df: pd.DataFrame) -> np.ndarray:
    """Encode risk_level column → 0=Thấp, 1=Trung bình, 2=Cao"""
    mapping = {"Thấp": 0, "Trung bình": 1, "Cao": 2}
    return df["risk_level"].map(mapping).fillna(1).astype(int).values


def prepare_plo_targets(df: pd.DataFrame) -> dict:
    """Return dict: plo → binary label array (1=pass, 0=fail), NaN → exclude."""
    targets = {}
    for plo in PLO_CODES:
        col = f"dat_{plo}"
        if col in df.columns:
            mask = df[col].notna()
            targets[plo] = {
                "y": df.loc[mask, col].astype(int).values,
                "mask": mask.values,
            }
    return targets
