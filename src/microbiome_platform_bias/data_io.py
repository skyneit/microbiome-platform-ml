from __future__ import annotations

from array import array
from pathlib import Path
from typing import Dict, Iterable, Tuple
import csv
import gzip

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

META_COLS = [
    'Sample_ID', 'Cohort', 'Group', 'Enterotype', 'Enterogroup',
    'EGroup', 'Platform', 'Subgroup', 'Status', 'y', 'ReadType'
]


def load_meta(path: str | Path, sample_col: str, group_col: str, platform_col: str,
              positive_label: str = 'D', negative_label: str = 'H',
              fl_platform_label: str = 'FL') -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {sample_col, group_col, platform_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'{path} missing required columns: {sorted(missing)}')

    df = df[df[group_col].isin([negative_label, positive_label])].copy()
    df['y'] = df[group_col].map({negative_label: 0, positive_label: 1}).astype(int)
    df['ReadType'] = np.where(
        df[platform_col].astype(str).str.upper().eq(fl_platform_label.upper()),
        'Long-read (FL)',
        'Short-read (VR)'
    )
    meta = df[[sample_col, platform_col, group_col, 'y', 'ReadType']].drop_duplicates()
    meta = meta.rename(columns={sample_col: 'Sample_ID', platform_col: 'Platform', group_col: 'Group'})
    return meta.set_index('Sample_ID')


def load_feature_csv(path: str | Path, meta: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'Sample_ID' not in df.columns:
        raise ValueError(f'{path} missing Sample_ID column')
    if 'Group' in df.columns:
        df = df[df['Group'].isin(['H', 'D'])].copy()

    df = df.set_index('Sample_ID')
    df = df.loc[df.index.intersection(meta.index)].copy()
    if df.empty:
        raise ValueError(f'{path}: no overlapping Sample_ID with metadata')

    if 'Enterotype' in df.columns and (df['Enterotype'].dtype == object or str(df['Enterotype'].dtype).startswith('string')):
        dummies = pd.get_dummies(df['Enterotype'], prefix='Enterotype', dtype=float)
        base = df.drop(columns=[c for c in META_COLS if c in df.columns], errors='ignore')
        X = pd.concat([base.select_dtypes(include=[np.number]).copy(), dummies], axis=1)
    else:
        base = df.drop(columns=[c for c in META_COLS if c in df.columns], errors='ignore')
        X = base.select_dtypes(include=[np.number]).copy()

    if X.shape[1] == 0:
        raise ValueError(f'{path}: no numeric features after dropping metadata columns')
    return X


def build_sparse_from_long(long_gz: str | Path, sample_ids: Iterable[str]) -> Tuple[csr_matrix, Dict[str, int]]:
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
