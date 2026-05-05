#!/usr/bin/env python
from __future__ import annotations

import argparse
from microbiome_platform_bias.config import load_config
from microbiome_platform_bias.model_compare_recalibration import run


def main() -> None:
    parser = argparse.ArgumentParser(description='Run single split model comparison with Platt and isotonic recalibration.')
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    run(cfg)


if __name__ == '__main__':
    main()
