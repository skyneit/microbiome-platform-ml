from __future__ import annotations

from array import array
from pathlib import Path
from typing import Dict, Iterable, Tuple
import csv
import gzip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.sparse import coo_matrix
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
import warnings

from .config import project_path
from .calibration import cox_calibration, calibration_in_the_large


def save_png(path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {path}')


def load_labels_from_ref(ref_csv: Path, group_col: str = 'Group') -> pd.DataFrame:
    df = pd.read_csv(ref_csv)
    if 'Sample_ID' not in df.columns:
        raise ValueError(f'{ref_csv} lacks Sample_ID')
    if group_col not in df.columns:
        raise ValueError(f'{ref_csv} lacks {group_col}')
    df = df[df[group_col].isin(['H', 'D'])].copy()
    df['y'] = df[group_col].map({'H': 0, 'D': 1}).astype(int)
    return df[['Sample_ID', 'y']].drop_duplicates().set_index('Sample_ID')


def load_numeric_X_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Group' in df.columns:
        df = df[df['Group'].isin(['H', 'D'])].copy()
    if 'Sample_ID' not in df.columns:
        raise ValueError(f'{path} lacks Sample_ID')
    df = df.set_index('Sample_ID')
    drop_cols = [c for c in ['Group', 'y'] if c in df.columns]
    return df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).copy()


def build_sparse_from_long(long_gz: Path, sample_ids: Iterable[str]):
    sample_ids = list(sample_ids)
    sample_to_i = {sid: i for i, sid in enumerate(sample_ids)}
    feat_to_j: Dict[str, int] = {}
    rows = array('I')
    cols = array('I')
    vals = array('f')
    with gzip.open(long_gz, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for sid, feat, value in reader:
            i = sample_to_i.get(sid)
            if i is None:
                continue
            j = feat_to_j.setdefault(feat, len(feat_to_j))
            rows.append(i)
            cols.append(j)
            vals.append(float(value))
    X = coo_matrix(
        (np.frombuffer(vals, dtype=np.float32),
         (np.frombuffer(rows, dtype=np.uint32), np.frombuffer(cols, dtype=np.uint32))),
        shape=(len(sample_ids), len(feat_to_j)),
        dtype=np.float32,
    ).tocsr()
    return X, feat_to_j


def fit_predict_xgb(X_train, y_train, X_test, xgb_params: Dict, random_state: int, n_jobs: int):
    params = dict(xgb_params)
    params.setdefault('random_state', random_state)
    params.setdefault('n_jobs', n_jobs)
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def fit_predict_xgb_calibrated(X_train, y_train, X_test, xgb_params: Dict, random_state: int,
                               n_jobs: int, method: str = 'sigmoid', cv: int = 5):
    params = dict(xgb_params)
    params.setdefault('random_state', random_state)
    params.setdefault('n_jobs', n_jobs)
    base = xgb.XGBClassifier(**params)
    try:
        cal = CalibratedClassifierCV(estimator=base, method=method, cv=cv)
    except TypeError:  # older scikit-learn
        cal = CalibratedClassifierCV(base_estimator=base, method=method, cv=cv)
    cal.fit(X_train, y_train)
    return cal.predict_proba(X_test)[:, 1]


def add_metrics_rows(metrics_rows: list, pred_df: pd.DataFrame, y_test, test_ids,
                     score_dict: Dict[str, np.ndarray], suffix: str, model_order: list[str]):
    for name in model_order:
        if name not in score_dict:
            continue
        y_score = np.asarray(score_dict[name], dtype=float)
        model_name = name if suffix == '' else f'{name}{suffix}'
        pred_df[f'{model_name}_proba'] = y_score
        y_pred = (y_score >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        metrics_rows.append({
            'Model': model_name,
            'n_train': np.nan,
            'n_test': len(test_ids),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'AUPRC': average_precision_score(y_test, y_score),
            'AUROC': auc(fpr, tpr),
            'Brier': brier_score_loss(y_test, y_score),
            'Cox_intercept': cox_calibration(y_test, y_score)[0],
            'Cox_slope': cox_calibration(y_test, y_score)[1],
            'CITL': calibration_in_the_large(y_test, y_score),
        })


def plot_roc(scores: Dict[str, np.ndarray], y_test, model_order, model_color, outpath: Path):
    plt.figure(figsize=(6, 6))
    for name in model_order:
        if name not in scores:
            continue
        fpr, tpr, _ = roc_curve(y_test, scores[name])
        au = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2.5, color=model_color.get(name, 'C0'), label=f'{name} (AUROC={au:.3f})')
    plt.plot([0, 1], [0, 1], '--', linewidth=1, color='0.5')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves (Uncalibrated)')
    plt.legend(loc='lower right', frameon=False)
    save_png(outpath)


def plot_pr(scores: Dict[str, np.ndarray], y_test, model_order, model_color, outpath: Path):
    plt.figure(figsize=(6, 6))
    for name in model_order:
        if name not in scores:
            continue
        precision, recall, _ = precision_recall_curve(y_test, scores[name])
        ap = average_precision_score(y_test, scores[name])
        plt.plot(recall, precision, linewidth=2.5, color=model_color.get(name, 'C0'), label=f'{name} (AUPRC={ap:.3f})')
    pos_rate = float(np.mean(y_test))
    plt.hlines(pos_rate, 0, 1, linestyles='--', linewidth=1, color='0.5', label=f'Baseline (pos rate={pos_rate:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves (Uncalibrated)')
    plt.legend(loc='lower left', frameon=False)
    save_png(outpath)


def plot_calibration_uncalibrated(scores: Dict[str, np.ndarray], y_test, model_order, model_color, outpath: Path):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], '--', linewidth=1, color='0.5', label='Perfectly calibrated')
    for name in model_order:
        if name not in scores:
            continue
        prob_true, prob_pred = calibration_curve(y_test, scores[name], n_bins=10, strategy='quantile')
        brier = brier_score_loss(y_test, scores[name])
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2.5, color=model_color.get(name, 'C0'),
                 label=f'{name} (Brier={brier:.3f})')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed event rate')
    plt.title('Calibration Curves (Uncalibrated)')
    plt.legend(loc='upper left', frameon=False, fontsize=9)
    save_png(outpath)


def plot_recalibration_selected(scores, scores_platt, scores_iso, y_test, cal_models, model_color, outpath: Path):
    plt.figure(figsize=(6.8, 6))
    plt.plot([0, 1], [0, 1], '--', linewidth=1, color='0.5', label='Perfectly calibrated')
    style_triplet = [(scores, '', '-'), (scores_platt, '+Platt', '--'), (scores_iso, '+Isotonic', ':')]
    for name in cal_models:
        if name not in scores:
            continue
        for score_dict, tag, linestyle in style_triplet:
            prob_true, prob_pred = calibration_curve(y_test, score_dict[name], n_bins=10, strategy='quantile')
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, linestyle=linestyle,
                     color=model_color.get(name, 'C0'), label=f'{name}{tag}')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Observed event rate')
    plt.title('Calibration: Uncalibrated vs Platt vs Isotonic (Selected Models)')
    plt.legend(loc='upper left', frameon=False, fontsize=9)
    save_png(outpath)


def run(cfg: Dict) -> None:
    random_state = int(cfg['project'].get('random_state', 42))
    n_jobs = int(cfg['project'].get('n_jobs', 8))
    xgb_params = cfg.get('xgboost', {})
    inputs = cfg['inputs']
    plotting = cfg['plotting']
    mc_cfg = cfg.get('model_compare_recalibration', {})

    outdir = project_path(cfg, mc_cfg.get('outdir', 'results/01_model_compare_recalibration'))
    outdir.mkdir(parents=True, exist_ok=True)

    model_order = plotting['model_order']
    model_color = plotting['model_color']
    cal_models = mc_cfg.get('calibration_models', ['Taxa-Genus', 'Taxa-Species', inputs.get('genus_function_name', 'GenusXPathway')])
    test_size = float(mc_cfg.get('test_size', 0.30))
    min_common_samples = int(mc_cfg.get('min_common_samples', 20))
    min_nnz = int(mc_cfg.get('min_nnz', cfg['analysis'].get('min_nnz_train', 5)))
    cv = int(mc_cfg.get('calibration_cv', 5))

    labels = load_labels_from_ref(project_path(cfg, inputs['reference_meta_csv']), group_col=cfg['analysis'].get('group_column', 'Group'))

    x_by_model = {}
    common_ids = set(labels.index)
    for name, path in inputs['csv_models'].items():
        X = load_numeric_X_from_csv(project_path(cfg, path))
        ids_here = set(X.index) & set(labels.index)
        x_by_model[name] = X.loc[sorted(ids_here)]
        common_ids &= ids_here

    common_ids = sorted(common_ids)
    if len(common_ids) < min_common_samples:
        raise ValueError(f'Too few common Sample_ID across CSV models: n={len(common_ids)}')

    y_all = labels.loc[common_ids, 'y'].values
    train_ids, test_ids = train_test_split(common_ids, test_size=test_size, random_state=random_state, stratify=y_all)
    train_ids = list(train_ids)
    test_ids = list(test_ids)
    y_train = labels.loc[train_ids, 'y'].values
    y_test = labels.loc[test_ids, 'y'].values

    scores: Dict[str, np.ndarray] = {}
    scores_platt: Dict[str, np.ndarray] = {}
    scores_iso: Dict[str, np.ndarray] = {}

    for name in inputs['csv_models'].keys():
        X = x_by_model[name]
        X_train = X.loc[train_ids]
        X_test = X.loc[test_ids]
        scores[name] = fit_predict_xgb(X_train, y_train, X_test, xgb_params, random_state, n_jobs)
        scores_platt[name] = fit_predict_xgb_calibrated(X_train, y_train, X_test, xgb_params, random_state, n_jobs, method='sigmoid', cv=cv)
        scores_iso[name] = fit_predict_xgb_calibrated(X_train, y_train, X_test, xgb_params, random_state, n_jobs, method='isotonic', cv=cv)

    gf_name = inputs.get('genus_function_name', 'GenusXPathway')
    needed_ids = train_ids + test_ids
    X_sparse_all, _ = build_sparse_from_long(project_path(cfg, inputs['genus_function_long']), needed_ids)
    n_train = len(train_ids)
    X_train_gf = X_sparse_all[:n_train]
    X_test_gf = X_sparse_all[n_train:]
    nnz_train = X_train_gf.getnnz(axis=0)
    keep = np.where(nnz_train >= min_nnz)[0]
    X_train_gf = X_train_gf[:, keep]
    X_test_gf = X_test_gf[:, keep]
    scores[gf_name] = fit_predict_xgb(X_train_gf, y_train, X_test_gf, xgb_params, random_state, n_jobs)
    scores_platt[gf_name] = fit_predict_xgb_calibrated(X_train_gf, y_train, X_test_gf, xgb_params, random_state, n_jobs, method='sigmoid', cv=cv)
    scores_iso[gf_name] = fit_predict_xgb_calibrated(X_train_gf, y_train, X_test_gf, xgb_params, random_state, n_jobs, method='isotonic', cv=cv)

    metrics_rows = []
    pred_df = pd.DataFrame({'Sample_ID': test_ids, 'y_true': y_test})
    add_metrics_rows(metrics_rows, pred_df, y_test, test_ids, scores, '', model_order)
    add_metrics_rows(metrics_rows, pred_df, y_test, test_ids, scores_platt, '+Platt', model_order)
    add_metrics_rows(metrics_rows, pred_df, y_test, test_ids, scores_iso, '+Isotonic', model_order)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df['n_train'] = len(train_ids)
    metrics_df.to_csv(outdir / 'performance_summary_with_recalibration.csv', index=False)
    pred_df.to_csv(outdir / 'test_predictions_with_recalibration.csv', index=False)
    print(f'Saved: {outdir / "performance_summary_with_recalibration.csv"}')
    print(f'Saved: {outdir / "test_predictions_with_recalibration.csv"}')

    plot_roc(scores, y_test, model_order, model_color, outdir / 'ROC_overlay_uncalibrated.png')
    plot_pr(scores, y_test, model_order, model_color, outdir / 'PR_overlay_uncalibrated.png')
    plot_calibration_uncalibrated(scores, y_test, model_order, model_color, outdir / 'Calibration_overlay_uncalibrated.png')
    plot_recalibration_selected(scores, scores_platt, scores_iso, y_test, cal_models, model_color, outdir / 'Calibration_recalibration_selected.png')
    print(f'Done. Outputs are in: {outdir}')
