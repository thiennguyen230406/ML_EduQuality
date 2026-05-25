"""
performance_model.py — Student Performance & Risk Model (VNUK DS 2020)
  1. RiskClassifier    → Thấp / Trung bình / Cao  (RF + XGB VotingClassifier)
  2. GPARegressor      → Predicted semester GPA    (XGBoost Regressor)
  3. DropoutDetector   → Binary dropout flag        (SMOTE + Random Forest)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score,
                              mean_absolute_error, r2_score, f1_score)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
from pathlib import Path
import sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RISK_LABELS    = {0: "Thấp", 1: "Trung bình", 2: "Cao"}
RISK_LABELS_VN = {"Thấp": 0, "Trung bình": 1, "Cao": 2}
RISK_COLORS    = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}

DROPOUT_ACTIONS = {
    0: "Tiếp tục theo dõi định kỳ",
    1: "Cần can thiệp ngay — liên hệ cố vấn học tập",
}


class PerformanceModel:
    """Unified model bundle for risk classification, GPA regression, dropout detection."""

    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds      = cv_folds
        self.random_state  = random_state
        self.risk_clf      = None
        self.gpa_reg       = None
        self.dropout_clf   = None
        self.risk_encoder  = LabelEncoder()
        self.metrics: dict = {}
        self.feature_names: list = []

    # ── Build estimators ──────────────────────────────────────────────────────

    def _build_risk_clf(self):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=self.random_state, n_jobs=-1
        )
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=self.random_state, verbosity=0
        )
        return VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb_clf)],
            voting="soft", n_jobs=-1
        )

    def _build_gpa_reg(self):
        return xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, verbosity=0
        )

    def _build_dropout_clf(self):
        return RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=self.random_state, n_jobs=-1
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, df: pd.DataFrame, feature_names: list = None):
        """
        X             : preprocessed feature matrix
        df            : raw DataFrame containing labels (risk_level, diem_gpa, dropout_risk)
        feature_names : list of feature names for SHAP
        """
        self.feature_names = feature_names or []
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)

        # ── 1. Risk Classifier ────────────────────────────────────────────────
        print("\n  Training Risk Classifier ...")
        y_risk_str = df["risk_level"].fillna("Trung bình")
        y_risk     = y_risk_str.map(RISK_LABELS_VN).fillna(1).astype(int).values
        self.risk_clf = self._build_risk_clf()
        risk_f1 = cross_val_score(self.risk_clf, X, y_risk, cv=cv,
                                  scoring="f1_weighted", n_jobs=-1)
        self.risk_clf.fit(X, y_risk)
        self.metrics["risk"] = {
            "cv_f1_weighted": round(float(risk_f1.mean()), 4),
            "cv_f1_std":      round(float(risk_f1.std()), 4),
        }
        print(f"    ✓ Risk Clf   F1={risk_f1.mean():.3f}±{risk_f1.std():.3f}")

        # ── 2. GPA Regressor ──────────────────────────────────────────────────
        print("  Training GPA Regressor ...")
        y_gpa   = df["diem_gpa"].fillna(df["diem_gpa"].mean()).values
        cv_reg  = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        self.gpa_reg = self._build_gpa_reg()
        gpa_r2 = cross_val_score(self.gpa_reg, X, y_gpa, cv=cv_reg, scoring="r2", n_jobs=-1)
        self.gpa_reg.fit(X, y_gpa)
        self.metrics["gpa"] = {
            "cv_r2":     round(float(gpa_r2.mean()), 4),
            "cv_r2_std": round(float(gpa_r2.std()), 4),
        }
        print(f"    ✓ GPA Reg    R²={gpa_r2.mean():.3f}±{gpa_r2.std():.3f}")

        # ── 3. Dropout Detector ───────────────────────────────────────────────
        print("  Training Dropout Detector ...")
        y_drop = df["dropout_risk"].fillna(0).astype(int).values

        # Only train if there's some positive cases
        if y_drop.sum() >= 5:
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=self.random_state, k_neighbors=3)
                Xs, ys = sm.fit_resample(X, y_drop)
            except Exception:
                Xs, ys = X, y_drop

            self.dropout_clf = self._build_dropout_clf()
            drop_auc = cross_val_score(self.dropout_clf, Xs, ys, cv=cv,
                                       scoring="roc_auc", n_jobs=-1)
            self.dropout_clf.fit(Xs, ys)
            self.metrics["dropout"] = {
                "cv_auc":     round(float(drop_auc.mean()), 4),
                "cv_auc_std": round(float(drop_auc.std()), 4),
            }
            print(f"    ✓ Dropout    AUC={drop_auc.mean():.3f}±{drop_auc.std():.3f}")
        else:
            print("    ⚠ Dropout: too few positive samples — skipped")

        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> dict:
        """Predict for a batch (rows of X). Returns dict of arrays."""
        result = {}

        if self.risk_clf is not None:
            risk_int = self.risk_clf.predict(X)
            risk_proba = self.risk_clf.predict_proba(X)
            result["risk_level"]     = [RISK_LABELS[r] for r in risk_int]
            result["risk_proba"]     = risk_proba.tolist()
            result["risk_color"]     = [RISK_COLORS[r] for r in risk_int]

        if self.gpa_reg is not None:
            gpa_pred = self.gpa_reg.predict(X)
            result["gpa_predicted"] = np.clip(gpa_pred, 0.0, 4.0).tolist()

        if self.dropout_clf is not None:
            drop_proba  = self.dropout_clf.predict_proba(X)[:, 1]
            drop_pred   = (drop_proba >= 0.5).astype(int)
            result["dropout_risk"]  = drop_pred.tolist()
            result["dropout_proba"] = drop_proba.tolist()
            result["dropout_action"]= [DROPOUT_ACTIONS[d] for d in drop_pred]

        return result

    def predict_single(self, X_row: np.ndarray) -> dict:
        """Single-sample prediction with friendly Vietnamese output."""
        res = self.predict(X_row.reshape(1, -1))
        out = {}

        if "risk_level" in res:
            out["muc_rui_ro"]       = res["risk_level"][0]
            out["mau_canh_bao"]     = res["risk_color"][0]
            proba = res["risk_proba"][0]
            classes = ["Thấp", "Trung bình", "Cao"]
            out["xac_suat_rui_ro"]  = {c: round(float(p), 3) for c,p in zip(classes, proba)}

        if "gpa_predicted" in res:
            out["gpa_du_bao"] = round(float(res["gpa_predicted"][0]), 2)

        if "dropout_risk" in res:
            out["nguy_co_bo_hoc"]     = bool(res["dropout_risk"][0])
            out["xac_suat_bo_hoc"]    = round(float(res["dropout_proba"][0]) * 100, 1)
            out["hanh_dong_de_xuat"]  = res["dropout_action"][0]

        return out

    def get_metrics(self) -> dict:
        return self.metrics

    def save(self, path: str = None):
        path = path or str(MODELS_DIR / "performance_model.pkl")
        joblib.dump(self, path)
        sz = Path(path).stat().st_size / 1024
        print(f"  ✓ PerformanceModel saved → {path}  ({sz:.1f} KB)")

    @staticmethod
    def load(path: str = None) -> "PerformanceModel":
        path = path or str(MODELS_DIR / "performance_model.pkl")
        return joblib.load(path)
