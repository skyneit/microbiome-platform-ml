from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from .modeling import safe_auc_possible


def save_roc_overlay(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str, out_png: str | Path,
                     plot_order: Iterable[str], color_map: Dict[str, str]) -> None:
    plt.figure(figsize=(6, 6))
    plotted = 0
    for model in [m for m in plot_order if m in curves]:
        y, p = curves[model]
        if not safe_auc_possible(y):
            continue
        fpr, tpr, _ = roc_curve(y, p)
        au = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2.5, color=color_map.get(model), label=f'{model} (AUROC={au:.3f})')
        plotted += 1
    plt.plot([0, 1], [0, 1], '--', lw=1, color='0.5')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    if plotted:
        plt.legend(frameon=False, loc='lower right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def save_shap_outputs(clf, X, feature_names, out_png: str | Path, out_csv: str | Path,
                      max_display: int = 30) -> None:
    import shap
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    shap_vals = np.asarray(shap_vals)

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    mean_raw = np.mean(shap_vals, axis=0)
    idx_top = np.argsort(mean_abs)[::-1][:max_display]

    pd.DataFrame({
        'rank': np.arange(1, len(idx_top) + 1),
        'feature': [feature_names[i] for i in idx_top],
        'mean_abs_shap': mean_abs[idx_top],
        'mean_shap': mean_raw[idx_top],
    }).to_csv(out_csv, index=False)

    X_plot = X[:, idx_top].toarray() if hasattr(X, 'tocsr') else np.asarray(X)[:, idx_top]
    plt.figure()
    shap.summary_plot(shap_vals[:, idx_top], X_plot,
                      feature_names=[feature_names[i] for i in idx_top],
                      show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
