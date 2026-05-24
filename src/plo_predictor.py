"""
plo_predictor.py — PLO Achievement Predictor (VNUK DS 2020)
Multi-label classifier: predicts probability of achieving each of 10 PLOs.
Ensemble: Random Forest + XGBoost + Logistic Regression (soft voting)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import joblib
from pathlib import Path
import sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.vnuk_schema import PLO_CODES, PLO_DEFINITIONS

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

ALERT_THRESHOLDS = {
    "PLO1": 0.50, "PLO2": 0.50, "PLO3": 0.50, "PLO4": 0.50, "PLO5": 0.50,
    "PLO6": 0.50, "PLO7": 0.50, "PLO8": 0.50, "PLO9": 0.50, "PLO10": 0.50,
}


class PLOPredictor:
    """
    Per-PLO binary classifier ensemble.
    Trains one VotingClassifier per PLO that has sufficient data.
    """

    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds     = cv_folds
        self.random_state = random_state
        self.models: dict = {}       # plo → fitted VotingClassifier
        self.thresholds   = ALERT_THRESHOLDS.copy()
        self.metrics: dict = {}
        self.trained_plos: list = []

    def _build_estimator(self):
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=self.random_state, n_jobs=-1
        )
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=self.random_state, verbosity=0
        )
        lr = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            random_state=self.random_state
        )
        return VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb_clf), ("lr", lr)],
            voting="soft", n_jobs=-1
        )

    def fit(self, X: np.ndarray, targets: dict, feature_names: list = None):
        """
        targets: dict from data_processor.prepare_plo_targets()
            {plo: {"y": labels, "mask": bool_array}}
        """
        self.feature_names = feature_names or []
        print("\n  Training PLO Predictor...")
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)

        for plo in PLO_CODES:
            if plo not in targets:
                continue
            info = targets[plo]
            mask = info["mask"]
            y    = info["y"]
            Xm   = X[mask]

            if len(Xm) < 50 or len(np.unique(y)) < 2:
                print(f"    ⚠ {plo}: skipped (insufficient samples or single class)")
                continue

            clf = self._build_estimator()
            try:
                aucs = cross_val_score(clf, Xm, y, cv=cv,
                                       scoring="roc_auc", n_jobs=-1)
                clf.fit(Xm, y)
                self.models[plo]  = clf
                self.trained_plos.append(plo)
                auc_mean = aucs.mean()
                self.metrics[plo] = {
                    "cv_auc": round(float(auc_mean), 4),
                    "cv_auc_std": round(float(aucs.std()), 4),
                    "n_samples": int(len(y)),
                    "pct_pass": round(float(y.mean()), 3),
                }
                print(f"    ✓ {plo}  AUC={auc_mean:.3f}±{aucs.std():.3f}  "
                      f"n={len(y)}  pass={y.mean():.1%}")
            except Exception as e:
                print(f"    ✗ {plo} training failed: {e}")

        return self

    def predict_proba(self, X: np.ndarray) -> dict:
        """Return {plo: probability_of_passing} for a sample (or batch)."""
        result = {}
        for plo, clf in self.models.items():
            proba = clf.predict_proba(X)
            # class index 1 = pass
            classes = list(clf.classes_)
            pass_idx = classes.index(1) if 1 in classes else 1
            result[plo] = proba[:, pass_idx]
        return result

    def predict_single(self, x: dict, feature_names: list) -> dict:
        """
        Predict PLO achievement for a single student record (dict).
        Returns structured result compatible with API.
        """
        from data.vnuk_schema import PLO_DEFINITIONS
        df = pd.DataFrame([x])
        # Ensure all required features present
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0.0
        X = df[feature_names].values.astype(float)
        probas = self.predict_proba(X)

        plo_results = {}
        alerts = []
        for plo in PLO_CODES:
            if plo not in probas:
                prob = None
                status = "Không đánh giá"
            else:
                prob = float(probas[plo][0])
                thr  = self.thresholds.get(plo, 0.5)
                if prob >= 0.75:
                    status = "Đạt tốt"
                elif prob >= thr:
                    status = "Đạt"
                else:
                    status = "Nguy cơ không đạt"
                    alerts.append({
                        "plo": plo,
                        "ten": PLO_DEFINITIONS[plo][:60],
                        "xac_suat": round(prob, 3),
                    })

            plo_results[plo] = {
                "xac_suat_dat": round(prob, 3) if prob is not None else None,
                "trang_thai": status,
                "mo_ta": PLO_DEFINITIONS.get(plo, ""),
            }

        # Overall PLO score
        valid = [v["xac_suat_dat"] for v in plo_results.values()
                 if v["xac_suat_dat"] is not None]
        overall = round(float(np.mean(valid)), 3) if valid else 0.0

        return {
            "plo_chi_tiet": plo_results,
            "diem_plo_tong": overall,
            "canh_bao_plo": alerts,
            "so_plo_nguy_co": len(alerts),
        }

    def get_metrics(self) -> dict:
        return self.metrics

    def save(self, path: str = None):
        path = path or str(MODELS_DIR / "plo_predictor.pkl")
        joblib.dump(self, path)
        sz = Path(path).stat().st_size / 1024
        print(f"  ✓ PLOPredictor saved → {path}  ({sz:.1f} KB)")

    @staticmethod
    def load(path: str = None) -> "PLOPredictor":
        path = path or str(MODELS_DIR / "plo_predictor.pkl")
        return joblib.load(path)
