from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from .modeling import choose_k_splits, safe_auc_possible, fit_xgb


def expit(x):
    x = np.asarray(x)
    return 1.0 / (1.0 + np.exp(-x))


def safe_logit(p, eps: float = 1e-15):
    p = np.clip(np.asarray(p), eps, 1 - eps)
    return np.log(p / (1 - p))


def _logistic_no_penalty(max_iter: int = 2000):
    try:
        return LogisticRegression(penalty=None, solver='lbfgs', max_iter=max_iter)
    except TypeError:  # older scikit-learn
        return LogisticRegression(penalty='none', solver='lbfgs', max_iter=max_iter)


def cox_calibration(y_true, y_prob, eps: float = 1e-15, max_iter: int = 2000) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    if not safe_auc_possible(y_true):
        return np.nan, np.nan
    z = safe_logit(y_prob, eps=eps).reshape(-1, 1)
    lr = _logistic_no_penalty(max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lr.fit(z, y_true)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def calibration_in_the_large(y_true, y_prob, eps: float = 1e-15, max_iter: int = 100, tol: float = 1e-10) -> float:
    y_true = np.asarray(y_true).astype(float)
    if not safe_auc_possible(y_true):
        return np.nan
    z = safe_logit(y_prob, eps=eps)
    a = 0.0
    for _ in range(max_iter):
        mu = expit(a + z)
        g = np.sum(y_true - mu)
        h = -np.sum(mu * (1 - mu))
        if h == 0:
            break
        step = g / h
        a -= step
        if abs(step) < tol:
            break
    return float(a)


def fit_xgb_and_recalibrate_return_base(X_train, y_train, X_test, params, random_state: int, n_jobs: int,
                                        sample_weight=None, inner_cv: int = 5, eps: float = 1e-15):
    """No-leakage recalibration.

    Inner CV is performed only within the training set to obtain OOF probabilities.
    Platt and isotonic calibrators are fit on those OOF probabilities, then applied
    to held-out-platform predictions from a model refit on the full training set.
    """
    y_train = np.asarray(y_train).astype(int)
    if not safe_auc_possible(y_train):
        p_const = float(np.mean(y_train))
        p = np.full(shape=(X_test.shape[0],), fill_value=p_const, dtype=float)
        return None, p, p.copy(), p.copy()

    k = choose_k_splits(y_train, max_splits=inner_cv)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    oof = np.zeros_like(y_train, dtype=float)

    for tr_i, va_i in skf.split(np.zeros(len(y_train)), y_train):
        sw_i = None if sample_weight is None else np.asarray(sample_weight)[tr_i]
        clf = fit_xgb(X_train[tr_i], y_train[tr_i], params=params, random_state=random_state, n_jobs=n_jobs)
        clf.fit(X_train[tr_i], y_train[tr_i], sample_weight=sw_i)
        oof[va_i] = clf.predict_proba(X_train[va_i])[:, 1]

    z_oof = safe_logit(oof, eps=eps).reshape(-1, 1)
    platt = _logistic_no_penalty(max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        platt.fit(z_oof, y_train, sample_weight=sample_weight)
    a_platt = float(platt.intercept_[0])
    b_platt = float(platt.coef_[0][0])

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    iso.fit(oof, y_train, sample_weight=sample_weight)

    base = fit_xgb(X_train, y_train, params=params, random_state=random_state, n_jobs=n_jobs)
    base.fit(X_train, y_train, sample_weight=sample_weight)
    p_raw = base.predict_proba(X_test)[:, 1]
    p_platt = expit(a_platt + b_platt * safe_logit(p_raw, eps=eps))
    p_iso = iso.transform(p_raw)
    return base, np.clip(p_raw, 0, 1), np.clip(p_platt, 0, 1), np.clip(p_iso, 0, 1)
