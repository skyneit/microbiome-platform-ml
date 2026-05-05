from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.model_selection import StratifiedKFold

from .config import project_path
from .data_io import build_sparse_from_long, load_feature_csv, load_meta
from .modeling import choose_k_splits, eval_binary, fit_xgb, safe_auc_possible
from .plots import save_roc_overlay, save_shap_outputs


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    k = min(r, c)
    return np.sqrt(chi2 / (n * (k - 1))) if k > 1 else np.nan


def row_subsample(X, y, max_n: int, seed: int):
    if len(y) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(y)), size=max_n, replace=False)
    return (X[idx] if hasattr(X, 'tocsr') else X[idx, :]), np.asarray(y)[idx]


class PlatformBiasAnalysis:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.outdir = project_path(cfg, cfg['project']['outdir'])
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.random_state = int(cfg['project']['random_state'])
        self.n_jobs = int(cfg['project']['n_jobs'])
        self.genusfunc_name = cfg['inputs']['genus_function_name']
        self.model_order = cfg['plotting']['model_order']
        self.model_color = cfg['plotting']['model_color']

        self.meta: Optional[pd.DataFrame] = None
        self.X_models: Dict[str, Optional[pd.DataFrame]] = {}
        self.avail_ids: Dict[str, set] = {}
        self.X_gf_all = None
        self.feat_names_gf_all: List[str] = []
        self.id2pos_sparse: Dict[str, int] = {}

    def load_inputs(self) -> None:
        a = self.cfg['analysis']
        self.meta = load_meta(
            project_path(self.cfg, self.cfg['inputs']['reference_meta_csv']),
            sample_col=a['sample_id_column'], group_col=a['group_column'], platform_col=a['platform_column'],
            positive_label=a['positive_label'], negative_label=a['negative_label'],
            fl_platform_label=a['fl_platform_label'],
        )

        for name, rel_path in self.cfg['inputs']['csv_models'].items():
            X = load_feature_csv(project_path(self.cfg, rel_path), self.meta)
            self.X_models[name] = X
            self.avail_ids[name] = set(X.index)

        sample_order = self.meta.index.tolist()
        self.X_gf_all, feat2j = build_sparse_from_long(
            project_path(self.cfg, self.cfg['inputs']['genus_function_long']), sample_order
        )
        self.feat_names_gf_all = [None] * len(feat2j)
        for feat, j in feat2j.items():
            self.feat_names_gf_all[j] = feat
        row_nnz = np.asarray(self.X_gf_all.getnnz(axis=1)).ravel()
        self.avail_ids[self.genusfunc_name] = {sid for sid, nnz in zip(sample_order, row_nnz) if nnz > 0}
        self.X_models[self.genusfunc_name] = None
        self.id2pos_sparse = {sid: i for i, sid in enumerate(sample_order)}

    def export_confounding_tables(self) -> None:
        meta = self.meta
        ct_plat = pd.crosstab(meta['Platform'], meta['Group'])
        chi2, p_plat, dof, _ = chi2_contingency(ct_plat.values)
        pd.DataFrame([{
            'test': 'Platform_by_Group', 'p_value': p_plat, 'chi2': chi2, 'dof': dof,
            'cramers_v': cramers_v(chi2, ct_plat.values.sum(), *ct_plat.shape)
        }]).to_csv(self.outdir / 'test_Platform_by_Group.csv', index=False)
        ct_plat.to_csv(self.outdir / 'crosstab_Platform_by_Group.csv')

        ct_rt = pd.crosstab(meta['ReadType'], meta['Group'])
        ct_rt.to_csv(self.outdir / 'crosstab_ReadType_by_Group.csv')
        if ct_rt.shape == (2, 2):
            odds_ratio, p_value = fisher_exact(ct_rt.values)
            pd.DataFrame([{'test': 'ReadType_by_Group', 'p_value': p_value, 'odds_ratio': odds_ratio}]).to_csv(
                self.outdir / 'test_ReadType_by_Group.csv', index=False
            )
        else:
            chi2_rt, p_rt, dof_rt, _ = chi2_contingency(ct_rt.values)
            pd.DataFrame([{'test': 'ReadType_by_Group', 'p_value': p_rt, 'chi2': chi2_rt, 'dof': dof_rt}]).to_csv(
                self.outdir / 'test_ReadType_by_Group.csv', index=False
            )

    def _matrix_for_ids(self, model_name: str, ids: List[str]):
        if model_name != self.genusfunc_name:
            return self.X_models[model_name].loc[ids].values.astype(np.float32), list(self.X_models[model_name].columns)
        pos = np.array([self.id2pos_sparse[sid] for sid in ids], dtype=int)
        return self.X_gf_all[pos], self.feat_names_gf_all

    def _filter_sparse_train_features(self, X_train, X_test, feature_names):
        min_nnz = int(self.cfg['analysis']['min_nnz_train'])
        nnz_train = np.asarray(X_train.getnnz(axis=0)).ravel()
        keep = np.where(nnz_train >= min_nnz)[0]
        return X_train[:, keep], X_test[:, keep], [feature_names[i] for i in keep]

    def _fit_predict(self, model_name: str, train_ids: List[str], test_ids: List[str], return_model: bool = False):
        y_train = self.meta.loc[train_ids, 'y'].values
        y_test = self.meta.loc[test_ids, 'y'].values
        X_train, feature_names = self._matrix_for_ids(model_name, train_ids)
        X_test, _ = self._matrix_for_ids(model_name, test_ids)
        if model_name == self.genusfunc_name:
            X_train, X_test, feature_names = self._filter_sparse_train_features(X_train, X_test, feature_names)
            if len(feature_names) == 0:
                return None
        clf = fit_xgb(X_train, y_train, self.cfg['xgboost'], self.random_state, self.n_jobs)
        p_test = clf.predict_proba(X_test)[:, 1]
        if return_model:
            return clf, X_train, X_test, feature_names, y_train, y_test, p_test
        return y_test, p_test

    def run_within_platform_cv(self) -> pd.DataFrame:
        rows = []
        max_splits = int(self.cfg['analysis']['max_cv_splits'])
        min_n = int(self.cfg['analysis']['min_within_platform_n'])
        for platform in sorted(self.meta['Platform'].unique()):
            ids_platform = set(self.meta.index[self.meta['Platform'] == platform])
            for model_name in self.model_order:
                ids = sorted(ids_platform & self.avail_ids.get(model_name, set()))
                if len(ids) < min_n:
                    continue
                y = self.meta.loc[ids, 'y'].values
                if not safe_auc_possible(y):
                    continue
                skf = StratifiedKFold(n_splits=choose_k_splits(y, max_splits), shuffle=True, random_state=self.random_state)
                metrics = []
                for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
                    train_ids = [ids[i] for i in train_idx]
                    test_ids = [ids[i] for i in test_idx]
                    pred = self._fit_predict(model_name, train_ids, test_ids)
                    if pred is None:
                        continue
                    y_test, p_test = pred
                    metrics.append(eval_binary(y_test, p_test))
                if metrics:
                    dfm = pd.DataFrame(metrics)
                    rows.append({
                        'Platform': platform, 'Model': model_name, 'n': len(ids), 'pos_rate': float(np.mean(y)),
                        'AUROC_mean': dfm['AUROC'].mean(), 'AUROC_sd': dfm['AUROC'].std(ddof=1),
                        'AUPRC_mean': dfm['AUPRC'].mean(), 'Brier_mean': dfm['Brier'].mean(),
                        'Accuracy_mean': dfm['Accuracy'].mean(), 'n_splits': skf.n_splits,
                    })
        out = pd.DataFrame(rows)
        out.to_csv(self.outdir / 'within_platform_CV_all_models.csv', index=False)
        return out

    def run_lopo(self) -> pd.DataFrame:
        rows = []
        min_test = int(self.cfg['analysis']['min_test_n_lopo'])
        min_train = int(self.cfg['analysis']['min_train_n_lopo'])
        for platform in sorted(self.meta['Platform'].unique()):
            test_all = set(self.meta.index[self.meta['Platform'] == platform])
            train_all = set(self.meta.index[self.meta['Platform'] != platform])
            for model_name in self.model_order:
                test_ids = sorted(test_all & self.avail_ids.get(model_name, set()))
                train_ids = sorted(train_all & self.avail_ids.get(model_name, set()))
                if len(test_ids) < min_test or len(train_ids) < min_train:
                    continue
                y_test = self.meta.loc[test_ids, 'y'].values
                y_train = self.meta.loc[train_ids, 'y'].values
                if not (safe_auc_possible(y_test) and safe_auc_possible(y_train)):
                    continue
                pred = self._fit_predict(model_name, train_ids, test_ids)
                if pred is None:
                    continue
                y_test, p_test = pred
                rows.append({'Holdout_Platform': platform, 'Model': model_name,
                             'n_train': len(train_ids), 'n_test': len(test_ids), **eval_binary(y_test, p_test)})
        out = pd.DataFrame(rows)
        out.to_csv(self.outdir / 'LOPO_results_all_models.csv', index=False)
        return out

    def run_readtype_negative_control(self) -> pd.DataFrame:
        rows = []
        y_rt_all = (self.meta['ReadType'] == 'Long-read (FL)').astype(int)
        if not safe_auc_possible(y_rt_all.values):
            return pd.DataFrame()
        for model_name in self.model_order:
            ids = sorted(set(self.meta.index) & self.avail_ids.get(model_name, set()))
            if len(ids) < 50:
                continue
            y = y_rt_all.loc[ids].values
            if not safe_auc_possible(y):
                continue
            skf = StratifiedKFold(n_splits=choose_k_splits(y, int(self.cfg['analysis']['max_cv_splits'])),
                                  shuffle=True, random_state=self.random_state)
            aucs = []
            for train_idx, test_idx in skf.split(np.zeros(len(ids)), y):
                train_ids = [ids[i] for i in train_idx]
                test_ids = [ids[i] for i in test_idx]
                X_train, feature_names = self._matrix_for_ids(model_name, train_ids)
                X_test, _ = self._matrix_for_ids(model_name, test_ids)
                if model_name == self.genusfunc_name:
                    X_train, X_test, feature_names = self._filter_sparse_train_features(X_train, X_test, feature_names)
                clf = fit_xgb(X_train, y[train_idx], self.cfg['xgboost'], self.random_state, self.n_jobs)
                p = clf.predict_proba(X_test)[:, 1]
                aucs.append(eval_binary(y[test_idx], p)['AUROC'])
            rows.append({'Model': model_name, 'n': len(ids), 'ReadType_AUROC_mean': float(np.mean(aucs)),
                         'ReadType_AUROC_sd': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                         'n_splits': skf.n_splits})
        out = pd.DataFrame(rows)
        out.to_csv(self.outdir / 'NegativeControl_ReadTypePred_all_models.csv', index=False)
        return out

    def save_lopo_plots_and_shap(self) -> None:
        shap_cfg = self.cfg['shap']
        shap_enable = bool(shap_cfg['enable'])
        shap_models = set(shap_cfg['models'])
        for platform in sorted(self.meta['Platform'].unique()):
            curves = {}
            test_all = set(self.meta.index[self.meta['Platform'] == platform])
            train_all = set(self.meta.index[self.meta['Platform'] != platform])
            for model_name in self.model_order:
                test_ids = sorted(test_all & self.avail_ids.get(model_name, set()))
                train_ids = sorted(train_all & self.avail_ids.get(model_name, set()))
                if len(test_ids) < int(self.cfg['analysis']['min_test_n_lopo']) or len(train_ids) < int(self.cfg['analysis']['min_train_n_lopo']):
                    continue
                if not (safe_auc_possible(self.meta.loc[test_ids, 'y'].values) and safe_auc_possible(self.meta.loc[train_ids, 'y'].values)):
                    continue
                fitted = self._fit_predict(model_name, train_ids, test_ids, return_model=True)
                if fitted is None:
                    continue
                clf, X_train, X_test, feature_names, y_train, y_test, p_test = fitted
                curves[model_name] = (y_test, p_test)
                if shap_enable and model_name in shap_models:
                    X_train_s, _ = row_subsample(X_train, y_train, int(shap_cfg['max_samples']), int(shap_cfg['seed']))
                    save_shap_outputs(
                        clf, X_train_s, feature_names,
                        self.outdir / f'SHAP_beeswarm_LOPO_holdout_{platform}_{model_name}_train.png',
                        self.outdir / f'SHAP_top_features_LOPO_holdout_{platform}_{model_name}_train.csv',
                        int(shap_cfg['max_display'])
                    )
                    X_test_s, _ = row_subsample(X_test, y_test, int(shap_cfg['max_samples']), int(shap_cfg['seed']) + 1)
                    save_shap_outputs(
                        clf, X_test_s, feature_names,
                        self.outdir / f'SHAP_beeswarm_LOPO_holdout_{platform}_{model_name}_test.png',
                        self.outdir / f'SHAP_top_features_LOPO_holdout_{platform}_{model_name}_test.csv',
                        int(shap_cfg['max_display'])
                    )
            if curves:
                save_roc_overlay(curves, f'LOPO ROC overlay (hold out {platform})',
                                 self.outdir / f'ROC_LOPO_holdout_{platform}_overlay.png',
                                 self.model_order, self.model_color)

    def run_all(self) -> None:
        self.load_inputs()
        self.export_confounding_tables()
        self.run_within_platform_cv()
        self.run_lopo()
        self.run_readtype_negative_control()
        self.save_lopo_plots_and_shap()
