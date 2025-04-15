"""General utility functions and constants for the project."""

import pickle
import random
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from loguru import logger
import time
            
from collections.abc import Callable
from importlib import import_module

import polars as pl


def seed_everything(seed: int) -> None:
    """Seed all components of the model.

    Parameters
    ----------
    seed: int
        Seed value to use

    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed)


def save_object_to_disk(obj: Any, save_path: str) -> None:
    """Save an object to disk using pickle.

    Parameters
    ----------
    obj: Any
        Object to save
    save_path: str
        Path to save the object

    """
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)
        print(f"File saved to disk: {save_path}")


def load_object_from_disk(load_path: str) -> Any:
    """Load an object from disk using pickle.

    Parameters
    ----------
    load_path: str
        Path to load the obje..utils import save_object_to_disk, seed_everythingct

    Returns
    -------
    Any
        Loaded object

    """
    with open(load_path, "rb") as f:
        return pickle.load(f)
    


def wait_for_workers(output_dir: str | Path, sleep_time: int = 2):
    time_slept = 0
    output_dir = Path(output_dir)
    while any(output_dir.glob(".*.parquet_cache/locks/*.json")):
        time.sleep(sleep_time)
        time_slept += sleep_time
        if time_slept > 30:
            logger.warning(f"Waiting for: {list(output_dir.glob('.*.parquet_cache/locks/*.json'))}")
            



def create_prefix_or_chain(prefixes: list[str]) -> pl.Expr:
    expr = pl.lit(False)
    for prefix in prefixes:
        expr = expr | pl.col("code").str.starts_with(prefix)
    return expr


def apply_vocab_to_multitoken_codes(
    df: pl.DataFrame, cols: list[str], vocab: list[str]
) -> pl.DataFrame:
    df = df.with_columns(pl.when(pl.col(col).is_in(vocab)).then(col).alias(col) for col in cols)
    for l_col, r_col in zip(cols, cols[1:]):
        df = df.with_columns(pl.when(pl.col(l_col).is_not_null()).then(r_col).alias(r_col))
    return df


def unify_code_names(col: pl.Expr) -> pl.Expr:
    return (
        col.str.to_uppercase().str.replace_all(r"[,.]", "").str.replace_all(" ", "_", literal=True)
    )


def static_class(cls):
    return cls()


def load_function(function_name: str, module_name: str) -> Callable:
    module = import_module(module_name)
    if "." in function_name:
        cls_name, function_name = function_name.split(".")
        module = getattr(module, cls_name)
    return getattr(module, function_name)