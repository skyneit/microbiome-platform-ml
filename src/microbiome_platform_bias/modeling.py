from __future__ import annotations

from typing import Dict
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score


def safe_auc_possible(y) -> bool:
    return len(np.unique(np.asarray(y).astype(int))) == 2


def choose_k_splits(y, max_splits: int = 5) -> int:
    y = np.asarray(y).astype(int)
    k = min(max_splits, int(np.sum(y == 0)), int(np.sum(y == 1)))
    return max(2, k)


def fit_xgb(X_train, y_train, params: Dict, random_state: int, n_jobs: int):
    params = dict(params)
    params.setdefault('random_state', random_state)
    params.setdefault('n_jobs', n_jobs)
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def eval_binary(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'AUROC': roc_auc_score(y_true, y_prob) if safe_auc_possible(y_true) else np.nan,
        'AUPRC': average_precision_score(y_true, y_prob) if safe_auc_possible(y_true) else np.nan,
        'Brier': brier_score_loss(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'n': len(y_true),
        'pos_rate': float(np.mean(y_true)),
    }
