"""
Experiment configuration processing.
"""
import os
import re
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .shared import (
    clean_str,
    is_nullish,
    normalize_key,
    parse_bool,
    parse_float,
    parse_int,
    parse_list,
    read_table,
)


class ExperimentConfig:
    SCHEMA: Dict[str, str] = {
        "experiment_path": "str",
        "ingredients_file": "str",
        "grow_threshold": "float",
        "nickname": "str",
        "batch_size": "int",
        "timeout_min": "int",
        "model_type": "int",
        "direction": "int",
        "beyond_frontier": "bool",
        "use_unique": "bool",
        "n_rollouts": "int",
        "n_bags": "int",
        "transfer_model_folder": "str",
        "transfer_rf_pkl": "str",
        "transfer_data_dir": "str",
        "redo_size": "int",
        "redo_threshold": "list_float",
        "aas_only": "bool",
        "separate_redos": "bool",
        "simulation_types": "list_int",
        "random_walk_increment": "int",
    }
    EXPECTED_KEYS = set(SCHEMA.keys())

    @staticmethod
    def _is_placeholder_header(value: Any) -> bool:
        """Return True when a cell/header looks like a pandas placeholder column name."""
        if value is None:
            return True
        s = str(value).strip().lower()
        if not s:
            return True
        return bool(re.match(r"^unnamed:\s*\d+$", s))

    @staticmethod
    def detect_format(df: pd.DataFrame) -> str:
        cols_norm = [normalize_key(c) for c in df.columns]
        horiz_score = sum(1 for c in cols_norm if c in ExperimentConfig.EXPECTED_KEYS)
        vert_score = 0
        if df.shape[1] >= 1:
            keys_in_first_col = [normalize_key(v) for v in df.iloc[:, 0].astype(str).tolist()]
            vert_score = len(set(keys_in_first_col) & ExperimentConfig.EXPECTED_KEYS)
        if vert_score > horiz_score:
            return "vertical"
        return "horizontal"

    @staticmethod
    def coerce_to_schema(partial: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, t in ExperimentConfig.SCHEMA.items():
            val = partial.get(key, None)
            if t == "str":
                v = clean_str(val)
                if v is not None and key in {"experiment_path", "transfer_model_folder", "transfer_rf_pkl", "transfer_data_dir"}:
                    v = os.path.expanduser(v)
                out[key] = v
            elif t == "int":
                out[key] = parse_int(val)
            elif t == "float":
                out[key] = parse_float(val)
            elif t == "bool":
                out[key] = parse_bool(val)
            elif t == "list_int":
                out[key] = parse_list(val, elem_type="int")
            elif t == "list_float":
                out[key] = parse_list(val, elem_type="float")
            else:
                out[key] = None if is_nullish(val) else val
        return out

    @staticmethod
    def build_from_vertical(df: pd.DataFrame) -> Dict[str, Any]:
        if df.shape[1] < 2:
            raise ValueError("Vertical format requires at least two columns (key, value).")
        key_col = df.columns[0]
        val_col = df.columns[1]

        vert: Dict[str, Any] = {}
        header_key = normalize_key(key_col)
        header_val = val_col
        if (
            header_key in ExperimentConfig.EXPECTED_KEYS
            and not is_nullish(header_val)
            and not ExperimentConfig._is_placeholder_header(header_val)
        ):
            vert[header_key] = header_val

        for _, row in df.iterrows():
            k_raw = row.get(key_col, None)
            v = row.get(val_col, None)
            if is_nullish(k_raw):
                continue
            k = normalize_key(k_raw)
            if k in ExperimentConfig.EXPECTED_KEYS:
                vert[k] = v

        if not vert:
            raise ValueError("No expected keys found in the first column for vertical format.")
        return ExperimentConfig.coerce_to_schema(vert)

    @staticmethod
    def build_from_horizontal(df: pd.DataFrame) -> List[Dict[str, Any]]:
        col_map = {c: normalize_key(c) for c in df.columns}
        df = df.rename(columns=col_map)
        keep_cols = [c for c in df.columns if c in ExperimentConfig.EXPECTED_KEYS]
        if not keep_cols:
            raise ValueError("No expected keys found in header for horizontal format.")
        df = df[keep_cols]

        experiments: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            if row.isna().all():
                continue
            partial = {k: row.get(k, None) for k in ExperimentConfig.EXPECTED_KEYS if k in row}
            cfg = ExperimentConfig.coerce_to_schema(partial)
            if not cfg.get("experiment_path"):
                continue
            experiments.append(cfg)
        if not experiments:
            raise ValueError("No valid experiments found (experiment_path missing or blank).")
        return experiments

    @staticmethod
    def parse_configs(input_path: str, sheet: Optional[Union[str, int]], force_format: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Read the input table and return a list of experiment config dicts.
        Vertical format yields a single config in the list; horizontal yields multiple.
        Allows configs without experiment_path.
        
        Args:
            input_path: Path to input file
            sheet: Excel sheet name or index
            force_format: Force "vertical" or "horizontal" format
        """
        df = read_table(input_path, sheet_name=sheet)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Failed to read a single DataFrame from input.")

        fmt = force_format or ExperimentConfig.detect_format(df)

        if fmt == "vertical":
            # Try the standard vertical builder first (it doesn't require experiment_path on its own).
            try:
                cfg = ExperimentConfig.build_from_vertical(df)
                return [cfg]
            except Exception:
                # Fallback: read first two columns as key-value pairs, keep only keys in SCHEMA.
                if df.shape[1] < 2:
                    raise ValueError("Vertical format requires at least two columns (key, value).")
                key_col = df.columns[0]
                val_col = df.columns[1]
                partial: Dict[str, Any] = {}
                for _, row in df.iterrows():
                    k_raw = row.get(key_col, None)
                    v = row.get(val_col, None)
                    if is_nullish(k_raw):
                        continue
                    k = normalize_key(k_raw)
                    if k in ExperimentConfig.SCHEMA:
                        partial[k] = v
                cfg = ExperimentConfig.coerce_to_schema(partial)
                return [cfg]

        # Horizontal: normalize headers and coerce per row, keep rows even if experiment_path is missing.
        col_map = {c: normalize_key(c) for c in df.columns}
        df = df.rename(columns=col_map)
        keep_cols = [c for c in df.columns if c in ExperimentConfig.SCHEMA]

        configs: List[Dict[str, Any]] = []
        if not keep_cols:
            # No known keys in header: produce empty-schema configs for non-empty rows.
            for _, row in df.iterrows():
                if row.isna().all():
                    continue
                configs.append(ExperimentConfig.coerce_to_schema({}))
            return configs

        df = df[keep_cols]
        for _, row in df.iterrows():
            if row.isna().all():
                continue
            partial = {k: row.get(k, None) for k in keep_cols}
            cfg = ExperimentConfig.coerce_to_schema(partial)
            configs.append(cfg)
        return configs
