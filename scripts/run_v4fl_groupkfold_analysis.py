#!/usr/bin/env python3
from __future__ import annotations

import argparse
from microbiome_platform_bias.config import load_config
from microbiome_platform_bias.groupkfold_v4fl import V4FLGroupKFoldAnalysis


def main() -> None:
    parser = argparse.ArgumentParser(description='Run V4-vs-FL platform-held-out GroupKFold analysis with no-leakage calibration and SHAP.')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to YAML configuration file.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    analysis = V4FLGroupKFoldAnalysis(cfg)
    analysis.run_all()
    print(f'Done. Outputs saved to: {analysis.outdir}')


if __name__ == '__main__':
    main()
