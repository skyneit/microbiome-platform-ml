from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML config and return a mutable dictionary."""
    config_path = Path(config_path)
    with config_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['_config_dir'] = str(config_path.parent.resolve())
    cfg['_project_root'] = str(config_path.parent.parent.resolve())
    return cfg


def project_path(cfg: Dict[str, Any], path: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(cfg['_project_root']) / p
