"""
Models Module: Blended Random Forest + XGBoost Ensemble
Amazon ML Challenge 2026: Business Entity Resolution
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np

class BlendedEnsemble:
    def __init__(self, rf_model=None, xgb_model=None, w_xgb: float = 0.50, w_rf: float = 0.50,
                 threshold: float = 0.82, margin_threshold: float = 0.10):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.w_xgb = w_xgb
        self.w_rf = w_rf
        self.threshold = threshold
        self.margin_threshold = margin_threshold

    @classmethod
    def load(cls, models_dir: Path):
        rf = joblib.load(models_dir / "random_forest_model.joblib")
        xgb = joblib.load(models_dir / "xgboost_model.joblib")
        with open(models_dir / "ensemble_config.json", "r") as f:
            cfg = json.load(f)
        return cls(
            rf_model=rf,
            xgb_model=xgb,
            w_xgb=cfg.get("w_xgb", 0.50),
            w_rf=cfg.get("w_rf", 0.50),
            threshold=cfg.get("threshold", 0.82),
            margin_threshold=cfg.get("margin_threshold", 0.10)
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        rf_p = self.rf_model.predict_proba(X)[:, 1]
        xgb_p = self.xgb_model.predict_proba(X)[:, 1]
        return (self.w_xgb * xgb_p) + (self.w_rf * rf_p)
