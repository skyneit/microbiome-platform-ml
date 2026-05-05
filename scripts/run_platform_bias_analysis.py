#!/usr/bin/env python3
from __future__ import annotations

import argparse
from microbiome_platform_bias.analysis import PlatformBiasAnalysis
from microbiome_platform_bias.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Run multi-model platform-bias, LOPO, ROC, and SHAP analyses.')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to YAML configuration file.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    analysis = PlatformBiasAnalysis(cfg)
    analysis.run_all()
    print(f'Done. Outputs saved to: {analysis.outdir}')


if __name__ == '__main__':
    main()
