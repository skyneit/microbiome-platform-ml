from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_curve, auc)
from sklearn.model_selection import GroupKFold

from .calibration import calibration_in_the_large, cox_calibration, fit_xgb_and_recalibrate_return_base
from .config import project_path
from .data_io import build_sparse_from_long, load_feature_csv
from .modeling import safe_auc_possible


# Compatibility helpers if using the older modeling.py from the first pipeline.
def _safe_roc_auc(y_true, y_prob):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_prob) if safe_auc_possible(y_true) else np.nan


def _safe_auprc(y_true, y_prob):
    return average_precision_score(y_true, y_prob) if safe_auc_possible(y_true) else np.nan


def savefig(path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def compute_weights_within_platform(platform, y):
    platform = np.asarray(platform)
    y = np.asarray(y).astype(int)
    w = np.ones_like(y, dtype=float)
    for p in np.unique(platform):
        idx = np.where(platform == p)[0]
        y_p = y[idx]
        n0 = np.sum(y_p == 0)
        n1 = np.sum(y_p == 1)
        if n0 == 0 or n1 == 0:
            continue
        w[idx[y_p == 0]] = 0.5 / n0
        w[idx[y_p == 1]] = 0.5 / n1
    return w / np.mean(w)


def match_downsample_within_platform(sample_ids, platform, y, seed: int = 42):
    rng = np.random.default_rng(seed)
    platform = np.asarray(platform)
    y = np.asarray(y).astype(int)
    keep_positions = []
    for p in np.unique(platform):
        idx = np.where(platform == p)[0]
        idx0 = idx[y[idx] == 0]
        idx1 = idx[y[idx] == 1]
        if len(idx0) == 0 or len(idx1) == 0:
            continue
        m = min(len(idx0), len(idx1))
        keep_positions.extend(rng.choice(idx0, size=m, replace=False).tolist())
        keep_positions.extend(rng.choice(idx1, size=m, replace=False).tolist())
    return sorted(keep_positions)


def evaluate_fold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    auprc = _safe_auprc(y_true, y_prob)
    baseline = float(np.mean(y_true))
    cox_int, cox_slope = cox_calibration(y_true, y_prob)
    return dict(
        AUROC=_safe_roc_auc(y_true, y_prob),
        AUPRC=auprc,
        AUPRC_baseline=baseline,
        AUPRC_lift=(auprc - baseline) if np.isfinite(auprc) else np.nan,
        Brier=brier_score_loss(y_true, y_prob),
        Cox_intercept=cox_int,
        Cox_slope=cox_slope,
        CITL=calibration_in_the_large(y_true, y_prob),
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        F1=f1_score(y_true, y_pred, zero_division=0),
        n=len(y_true),
    )


def _sample_indices(n: int, max_n: int, seed: int):
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(n), size=max_n, replace=False))


def shap_beeswarm_and_rank_dense(model, X_test_np, feature_names, out_prefix: Path, max_samples: int, max_display: int,
                                 seed: int):
    import shap
    idx = _sample_indices(X_test_np.shape[0], max_samples, seed=seed)
    Xs = X_test_np[idx]
    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(Xs))
    mean_abs = np.mean(np.abs(sv), axis=0)
    rank = np.argsort(mean_abs)[::-1]
    rank_df = pd.DataFrame({'Feature': np.asarray(feature_names)[rank], 'MeanAbsSHAP': mean_abs[rank]})
    rank_df.to_csv(str(out_prefix) + '_shap_rank.csv', index=False)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, Xs, feature_names=feature_names, max_display=max_display, show=False)
    savefig(Path(str(out_prefix) + '_shap_beeswarm.png'))
    return rank_df


def shap_beeswarm_and_rank_sparse(model, X_test_sparse, feature_names_subset, out_prefix: Path, max_samples: int,
                                  max_display: int, sparse_bee_topk: int, seed: int):
    import shap
    idx = _sample_indices(X_test_sparse.shape[0], max_samples, seed=seed)
    Xs = X_test_sparse[idx]
    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(Xs))
    mean_abs = np.mean(np.abs(sv), axis=0)
    rank = np.argsort(mean_abs)[::-1]
    rank_df = pd.DataFrame({'Feature': np.asarray(feature_names_subset)[rank], 'MeanAbsSHAP': mean_abs[rank]})
    rank_df.to_csv(str(out_prefix) + '_shap_rank.csv', index=False)

    topk = rank[:min(sparse_bee_topk, len(rank))]
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv[:, topk], Xs[:, topk].toarray(),
                      feature_names=np.asarray(feature_names_subset)[topk].tolist(),
                      max_display=min(max_display, len(topk)), show=False)
    savefig(Path(str(out_prefix) + '_shap_beeswarm.png'))
    return rank_df


class V4FLGroupKFoldAnalysis:
    """V4-vs-FL platform-held-out GroupKFold analysis with no-leakage calibration."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.random_state = int(cfg['project']['random_state'])
        self.n_jobs = int(cfg['project']['n_jobs'])
        gcfg = cfg['v4fl_groupkfold']
        self.outdir = project_path(cfg, gcfg['outdir'])
        self.figdir = self.outdir / 'figures'
        self.shapdir = self.outdir / 'shap'
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.figdir.mkdir(parents=True, exist_ok=True)
        self.shapdir.mkdir(parents=True, exist_ok=True)
        self.gcfg = gcfg
        self.genusfunc_name = cfg['inputs']['genus_function_name']
        self.model_order = cfg['plotting']['model_order']
        self.model_color = cfg['plotting']['model_color']

    def load_meta_v4fl(self) -> pd.DataFrame:
        a = self.cfg['analysis']
        df = pd.read_csv(project_path(self.cfg, self.cfg['inputs']['reference_meta_csv']))
        required = {a['sample_id_column'], a['group_column'], a['platform_column']}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f'reference metadata missing columns: {sorted(missing)}')
        df = df[df[a['group_column']].isin([a['negative_label'], a['positive_label']])].copy()
        df = df[df[a['platform_column']].isin(self.gcfg['platforms_keep'])].copy()
        df['y'] = df[a['group_column']].map({a['negative_label']: 0, a['positive_label']: 1}).astype(int)
        out = df[[a['sample_id_column'], a['platform_column'], 'y']].drop_duplicates()
        return out.rename(columns={a['sample_id_column']: 'Sample_ID', a['platform_column']: 'Platform'}).set_index('Sample_ID')

    def should_do_shap(self, model_name: str) -> bool:
        scfg = self.cfg.get('shap', {})
        if not scfg.get('enable', True):
            return False
        models = self.gcfg.get('shap_models', 'all')
        return models == 'all' or model_name in models

    def plot_roc_overlay_raw(self, pred_df: pd.DataFrame, tag: str):
        plt.figure(figsize=(6, 6))
        for m in self.model_order:
            d = pred_df[pred_df['Model'] == m]
            if d.empty or not safe_auc_possible(d['y_true'].values):
                continue
            fpr, tpr, _ = roc_curve(d['y_true'].values, d['y_proba_raw'].values)
            au = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2.5, color=self.model_color.get(m, 'C0'), label=f'{m} (AUROC={au:.3f})')
        plt.plot([0, 1], [0, 1], '--', lw=1, color='0.5')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC (raw) — {tag}')
        plt.legend(loc='lower right', frameon=False)
        savefig(self.figdir / f'ROC_raw_{tag}.png')

    def plot_pr_overlay_raw(self, pred_df: pd.DataFrame, tag: str):
        plt.figure(figsize=(6, 6))
        baseline = float(pred_df['y_true'].mean())
        for m in self.model_order:
            d = pred_df[pred_df['Model'] == m]
            if d.empty or not safe_auc_possible(d['y_true'].values):
                continue
            precision, recall, _ = precision_recall_curve(d['y_true'].values, d['y_proba_raw'].values)
            ap = average_precision_score(d['y_true'].values, d['y_proba_raw'].values)
            plt.plot(recall, precision, lw=2.5, color=self.model_color.get(m, 'C0'), label=f'{m} (AUPRC={ap:.3f})')
        plt.hlines(baseline, 0, 1, linestyles='--', lw=1, color='0.5', label=f'Baseline (prev={baseline:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR (raw) — {tag}')
        plt.legend(loc='lower left', frameon=False)
        savefig(self.figdir / f'PR_raw_{tag}.png')

    def plot_calibration_overlay(self, pred_df: pd.DataFrame, calib: str, tag: str, n_bins: int = 10):
        col = f'y_proba_{calib}'
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], '--', lw=1, color='0.5', label='Perfectly calibrated')
        for m in self.model_order:
            d = pred_df[pred_df['Model'] == m]
            if d.empty or not safe_auc_possible(d['y_true'].values):
                continue
            pt, pp = calibration_curve(d['y_true'].values, d[col].values, n_bins=n_bins, strategy='quantile')
            b = brier_score_loss(d['y_true'].values, d[col].values)
            plt.plot(pp, pt, marker='o', lw=2.5, color=self.model_color.get(m, 'C0'), label=f'{m} (Brier={b:.3f})')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Observed event rate')
        plt.title(f'Calibration ({calib}) — {tag}')
        plt.legend(frameon=False, loc='upper left')
        savefig(self.figdir / f'CalibrationOverlay_{calib}_{tag}.png')

    def plot_auprc_lift_bar(self, metrics_df: pd.DataFrame, calib: str = 'raw'):
        d = metrics_df[metrics_df['Calibration'] == calib].copy()
        if d.empty:
            return
        agg = d.groupby('Model', as_index=False).agg(
            AUPRC_lift_mean=('AUPRC_lift', 'mean'), AUPRC_lift_sd=('AUPRC_lift', 'std'), folds=('Fold', 'nunique')
        )
        agg['Model'] = pd.Categorical(agg['Model'], categories=self.model_order, ordered=True)
        agg = agg.sort_values('Model')
        x = np.arange(len(agg))
        plt.figure(figsize=(8, 4.5))
        plt.bar(x, agg['AUPRC_lift_mean'].values,
                yerr=np.nan_to_num(agg['AUPRC_lift_sd'].values, nan=0.0), capsize=4,
                color=[self.model_color.get(str(m), 'C0') for m in agg['Model']])
        plt.axhline(0, lw=1, color='0.5')
        plt.xticks(x, agg['Model'].astype(str).tolist(), rotation=30, ha='right')
        plt.ylabel('AUPRC lift (AUPRC − prevalence)')
        plt.title(f'AUPRC lift across held-out platforms (mean±SD) [{calib}]')
        savefig(self.figdir / f'AUPRC_lift_bar_V4FL_GroupKFold_{calib}.png')

    def run_all(self):
        np.random.seed(self.random_state)
        meta = self.load_meta_v4fl()
        csv_models = self.cfg['inputs']['csv_models']
        common_ids = set(meta.index)
        X_csv_df = {}
        for name, path in csv_models.items():
            Xdf = load_feature_csv(project_path(self.cfg, path), meta)
            ids_here = set(Xdf.index) & set(meta.index)
            X_csv_df[name] = Xdf.loc[sorted(ids_here)]
            common_ids &= ids_here

        common_ids = sorted(common_ids)
        if len(common_ids) < int(self.gcfg.get('min_common_samples', 30)):
            raise ValueError(f'Too few common Sample_ID across CSV models in V4/FL: n={len(common_ids)}')

        platform_all = meta.loc[common_ids, 'Platform'].values
        y_all = meta.loc[common_ids, 'y'].values

        if self.gcfg.get('balance_method', 'weights') == 'match':
            keep_pos = match_downsample_within_platform(common_ids, platform_all, y_all, seed=self.random_state)
            common_ids = [common_ids[i] for i in keep_pos]
            platform_all = platform_all[keep_pos]
            y_all = y_all[keep_pos]
            for k in list(X_csv_df.keys()):
                X_csv_df[k] = X_csv_df[k].loc[common_ids]

        X_csv_np = {}
        X_csv_names = {}
        for k, Xdf in X_csv_df.items():
            Xk = Xdf.loc[common_ids]
            X_csv_np[k] = Xk.values.astype(np.float32)
            X_csv_names[k] = Xk.columns.tolist()

        X_gf_all, feat2j = build_sparse_from_long(project_path(self.cfg, self.cfg['inputs']['genus_function_long']), common_ids)
        gf_feature_names = [''] * len(feat2j)
        for feat, j in feat2j.items():
            gf_feature_names[j] = feat

        row_nnz = np.asarray(X_gf_all.getnnz(axis=1)).ravel()
        if np.any(row_nnz == 0):
            keep = np.where(row_nnz > 0)[0]
            common_ids = [common_ids[i] for i in keep]
            platform_all = platform_all[keep]
            y_all = y_all[keep]
            for k in X_csv_np:
                X_csv_np[k] = X_csv_np[k][keep]
            X_gf_all = X_gf_all[keep]

        groups = platform_all
        gkf = GroupKFold(n_splits=len(np.unique(groups)))
        fold_metrics: List[Dict] = []
        pred_rows: List[Dict] = []
        shap_rank_rows = []
        xgb_params = self.cfg['xgboost']
        shap_cfg = self.cfg.get('shap', {})

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(np.zeros(len(common_ids)), y_all, groups=groups), start=1):
            plat_tr = groups[tr_idx]
            plat_te = groups[te_idx]
            y_tr = y_all[tr_idx]
            y_te = y_all[te_idx]
            fold_tag = f"Fold{fold}_train={'+'.join(sorted(set(plat_tr)))}_test={'+'.join(sorted(set(plat_te)))}"
            sw = compute_weights_within_platform(plat_tr, y_tr) if self.gcfg.get('balance_method', 'weights') == 'weights' else None
            te_ids = [common_ids[i] for i in te_idx]

            for model_name in csv_models.keys():
                X_all = X_csv_np[model_name]
                X_tr = X_all[tr_idx]
                X_te = X_all[te_idx]
                base, p_raw, p_platt, p_iso = fit_xgb_and_recalibrate_return_base(
                    X_tr, y_tr, X_te, params=xgb_params, random_state=self.random_state, n_jobs=self.n_jobs,
                    sample_weight=sw, inner_cv=int(self.gcfg.get('inner_cv', 5))
                )
                for calib_name, p in [('raw', p_raw), ('platt', p_platt), ('isotonic', p_iso)]:
                    fold_metrics.append({'Fold': fold, 'FoldTag': fold_tag, 'Model': model_name,
                                         'Balance': self.gcfg.get('balance_method', 'weights'),
                                         'Calibration': calib_name, 'n_train': len(tr_idx), 'n_test': len(te_idx),
                                         **evaluate_fold(y_te, p)})
                for sid, yt, pr, pp, pi, plat in zip(te_ids, y_te, p_raw, p_platt, p_iso, plat_te):
                    pred_rows.append({'Fold': fold, 'FoldTag': fold_tag, 'Model': model_name, 'Sample_ID': sid,
                                      'Platform': plat, 'y_true': int(yt), 'y_proba_raw': float(pr),
                                      'y_proba_platt': float(pp), 'y_proba_isotonic': float(pi)})
                if self.should_do_shap(model_name) and base is not None:
                    rank_df = shap_beeswarm_and_rank_dense(
                        base, X_te, X_csv_names[model_name], self.shapdir / f'{model_name}_{fold_tag}',
                        max_samples=int(shap_cfg.get('max_samples', 300)),
                        max_display=int(shap_cfg.get('max_display', 30)), seed=self.random_state
                    )
                    rank_df['Model'] = model_name; rank_df['Fold'] = fold; rank_df['FoldTag'] = fold_tag
                    shap_rank_rows.append(rank_df)

            X_tr_gf = X_gf_all[tr_idx]
            X_te_gf = X_gf_all[te_idx]
            nnz_train = np.asarray(X_tr_gf.getnnz(axis=0)).ravel()
            keep_feat = np.where(nnz_train >= int(self.cfg['analysis']['min_nnz_train']))[0]
            X_tr_gf2 = X_tr_gf[:, keep_feat]
            X_te_gf2 = X_te_gf[:, keep_feat]
            gf_feat_names = [gf_feature_names[j] for j in keep_feat]
            base, p_raw, p_platt, p_iso = fit_xgb_and_recalibrate_return_base(
                X_tr_gf2, y_tr, X_te_gf2, params=xgb_params, random_state=self.random_state, n_jobs=self.n_jobs,
                sample_weight=sw, inner_cv=int(self.gcfg.get('inner_cv', 5))
            )
            for calib_name, p in [('raw', p_raw), ('platt', p_platt), ('isotonic', p_iso)]:
                fold_metrics.append({'Fold': fold, 'FoldTag': fold_tag, 'Model': self.genusfunc_name,
                                     'Balance': self.gcfg.get('balance_method', 'weights'), 'Calibration': calib_name,
                                     'n_train': len(tr_idx), 'n_test': len(te_idx), **evaluate_fold(y_te, p)})
            for sid, yt, pr, pp, pi, plat in zip(te_ids, y_te, p_raw, p_platt, p_iso, plat_te):
                pred_rows.append({'Fold': fold, 'FoldTag': fold_tag, 'Model': self.genusfunc_name, 'Sample_ID': sid,
                                  'Platform': plat, 'y_true': int(yt), 'y_proba_raw': float(pr),
                                  'y_proba_platt': float(pp), 'y_proba_isotonic': float(pi)})
            if self.should_do_shap(self.genusfunc_name) and base is not None:
                rank_df = shap_beeswarm_and_rank_sparse(
                    base, X_te_gf2, gf_feat_names, self.shapdir / f'{self.genusfunc_name}_{fold_tag}',
                    max_samples=int(shap_cfg.get('max_samples', 300)), max_display=int(shap_cfg.get('max_display', 30)),
                    sparse_bee_topk=int(self.gcfg.get('sparse_bee_topk', 50)), seed=self.random_state
                )
                rank_df['Model'] = self.genusfunc_name; rank_df['Fold'] = fold; rank_df['FoldTag'] = fold_tag
                shap_rank_rows.append(rank_df)

        metrics_df = pd.DataFrame(fold_metrics)
        pred_df = pd.DataFrame(pred_rows)
        metrics_df.to_csv(self.outdir / 'per_fold_metrics.csv', index=False)
        pred_df.to_csv(self.outdir / 'predictions_by_fold.csv', index=False)
        summary = metrics_df.groupby(['Model', 'Balance', 'Calibration'], as_index=False).agg({
            'AUROC': ['mean', 'std'], 'AUPRC': ['mean', 'std'], 'AUPRC_baseline': ['mean', 'std'],
            'AUPRC_lift': ['mean', 'std'], 'Brier': ['mean', 'std'], 'CITL': ['mean', 'std'],
            'Cox_intercept': ['mean', 'std'], 'Cox_slope': ['mean', 'std'], 'Accuracy': ['mean', 'std'],
            'Precision': ['mean', 'std'], 'Recall': ['mean', 'std'], 'F1': ['mean', 'std'], 'n': ['mean', 'std']
        })
        summary.columns = ['_'.join([c for c in col if c]).rstrip('_') if isinstance(col, tuple) else col for col in summary.columns]
        summary.to_csv(self.outdir / 'summary_mean_sd.csv', index=False)

        for fold in sorted(pred_df['Fold'].unique()):
            df_f = pred_df[pred_df['Fold'] == fold].copy()
            self.plot_roc_overlay_raw(df_f, tag=f'fold{fold}')
            self.plot_pr_overlay_raw(df_f, tag=f'fold{fold}')
            for calib in ['raw', 'platt', 'isotonic']:
                self.plot_calibration_overlay(df_f, calib=calib, tag=f'fold{fold}')
        self.plot_roc_overlay_raw(pred_df, tag='pooled')
        self.plot_pr_overlay_raw(pred_df, tag='pooled')
        for calib in ['raw', 'platt', 'isotonic']:
            self.plot_calibration_overlay(pred_df, calib=calib, tag='pooled')
        self.plot_auprc_lift_bar(metrics_df, calib='raw')

        if shap_rank_rows:
            shap_all = pd.concat(shap_rank_rows, ignore_index=True)
            pooled = shap_all.groupby(['Model', 'Feature'], as_index=False).agg(
                MeanAbsSHAP_mean=('MeanAbsSHAP', 'mean'), MeanAbsSHAP_sd=('MeanAbsSHAP', 'std'), folds=('Fold', 'nunique')
            )
            pooled.to_csv(self.shapdir / 'pooled_shap_rank_mean_over_folds.csv', index=False)
            for m in pooled['Model'].unique():
                pooled[pooled['Model'] == m].sort_values('MeanAbsSHAP_mean', ascending=False).head(50).to_csv(
                    self.shapdir / f'top50_{m}_pooled.csv', index=False
                )
        return metrics_df, pred_df
