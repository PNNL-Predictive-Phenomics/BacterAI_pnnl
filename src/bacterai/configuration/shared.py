"""
Shared utilities for configuration parsing.
"""
import ast
import os
import re
from typing import Any, List, Optional, Union

import pandas as pd


# ---------- Shared helpers ----------
NA_STRINGS = {"na", "nan", "n/a", "", "null", "none"}


def is_nullish(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    if isinstance(x, str) and x.strip().lower() in NA_STRINGS:
        return True
    return False


def clean_str(x: Any) -> Optional[str]:
    if is_nullish(x):
        return None
    s = str(x).strip()
    return None if s.lower() in NA_STRINGS else s


def normalize_key(k: Any) -> str:
    s = str(k).strip().lower()
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def read_table(path: str, sheet_name: Optional[Union[str, int]] = None) -> pd.DataFrame:
    """
    Read CSV/TSV or Excel into a DataFrame. If initial read yields
    undecidable orientation, try a header-less read fallback.
    """
    ext = os.path.splitext(path)[1].lower()

    def _read(header="infer"):
        if ext in [".xlsx", ".xls"]:
            sn = 0 if sheet_name is None else sheet_name
            return pd.read_excel(path, sheet_name=sn, header=0 if header == "infer" else None)
        elif ext in [".csv", ".tsv"]:
            sep = "," if ext == ".csv" else "\t"
            return pd.read_csv(path, sep=sep, header=0 if header == "infer" else None)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    df = _read(header="infer")
    if isinstance(df, dict):
        first_key = next(iter(df))
        df = df[first_key]
    if df is None or df.shape[0] == 0:
        df2 = _read(header=None)
        if isinstance(df2, dict):
            first_key = next(iter(df2))
            df2 = df2[first_key]
        if df2 is not None and df2.shape[0] > 0:
            df = df2
    return df


def parse_bool(x: Any) -> Optional[bool]:
    if is_nullish(x):
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return True
    if s in {"false", "f", "no", "n", "0"}:
        return False
    return None


def parse_int(x: Any) -> Optional[int]:
    if is_nullish(x):
        return None
    try:
        f = float(str(x).strip())
        return int(round(f))
    except Exception:
        return None


def parse_float(x: Any) -> Optional[float]:
    if is_nullish(x):
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


def parse_list(x: Any, elem_type: str) -> Optional[List[Any]]:
    if is_nullish(x):
        return None
    if isinstance(x, list):
        return [_coerce_elem(e, elem_type) for e in x]
    s = str(x).strip()
    if not s:
        return None
    if s.startswith("[") or s.startswith("("):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return [_coerce_elem(e, elem_type) for e in val]
        except Exception:
            pass
    parts = [p.strip() for p in s.split(",")]
    return [_coerce_elem(p, elem_type) for p in parts if p != ""]


def _coerce_elem(e: Any, elem_type: str) -> Any:
    if elem_type == "int":
        v = parse_int(e)
        return v if v is not None else None
    if elem_type == "float":
        v = parse_float(e)
        return v if v is not None else None
    return e


def unit_normalize(u: Any) -> Optional[str]:
    s = clean_str(u)
    if s is None:
        return None
    if s.strip().lower() == "x":
        return "1X_dilution"
    return s


def to_float(x: Any) -> Optional[float]:
    return parse_float(x)


def to_int_maybe(x: Any) -> Optional[int]:
    v = to_float(x)
    if v is None:
        return None
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    return v  # float if not an exact integer


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    return df
