"""Utility modules for EHR-FM."""

from .artifact_loader import load_env, resolve_model_path, download_checkpoint

import importlib.util
from pathlib import Path

_parent_utils_path = Path(__file__).parent.parent / "utils.py"
if _parent_utils_path.exists():
    spec = importlib.util.spec_from_file_location("utils_module", _parent_utils_path)
    utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_module)
    create_prefix_or_chain = utils_module.create_prefix_or_chain
    unify_code_names = utils_module.unify_code_names
    static_class = utils_module.static_class
    wait_for_workers = utils_module.wait_for_workers
    load_function = utils_module.load_function
    apply_vocab_to_multitoken_codes = utils_module.apply_vocab_to_multitoken_codes
else:
    raise ImportError(f"Could not find utils.py at {_parent_utils_path}")

__all__ = [
    "load_env",
    "resolve_model_path",
    "download_checkpoint",
    "create_prefix_or_chain",
    "unify_code_names",
    "static_class",
    "wait_for_workers",
    "load_function",
    "apply_vocab_to_multitoken_codes",
]

