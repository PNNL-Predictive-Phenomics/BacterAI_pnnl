"""
Core logic for experiment and ingredient configuration processing.

Dependencies:
  pip install pandas openpyxl
"""
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

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


# ---------- Experiment config ----------
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
                if v is not None and key in {"experiment_path", "transfer_model_folder", "transfer_data_dir"}:
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
        if header_key in ExperimentConfig.EXPECTED_KEYS and not is_nullish(header_val):
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
        """
        df = read_table(input_path, sheet_name=sheet)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Failed to read a single DataFrame from input.")

        fmt = force_format or ExperimentConfig.detect_format(df)
        if fmt == "vertical":
            cfg = ExperimentConfig.build_from_vertical(df)
            if not cfg.get("experiment_path"):
                raise ValueError("experiment_path is required but missing in vertical format.")
            return [cfg]
        else:
            return ExperimentConfig.build_from_horizontal(df)
    
    @staticmethod
    def parse_configs_lenient(
        input_path: str,
        sheet: Optional[Union[str, int]],
        force_format: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Lenient experiment parsing:
        - Vertical: produce a single config even if experiment_path is missing
          (fallback to first two columns as key/value pairs if needed).
        - Horizontal: coerce each row to schema even if experiment_path is missing.
        Returns a list of config dicts (may contain entries without experiment_path).
        """
        df = read_table(input_path, sheet_name=sheet)
        if not hasattr(df, "shape"):
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


# ---------- Ingredients config ----------
class IngredientsConfig:
    @staticmethod
    def detect_format(df: pd.DataFrame) -> int:
        cols = set(df.columns)
        fmt2_keys = {"id", "ingredient", "type", "nominal_value", "min_value", "max_value"}
        fmt1_keys = {"condition", "reagent", "type", "nominal_value", "min_value", "max_value"}
        if fmt2_keys.issubset(cols):
            return 2
        if fmt1_keys.issubset(cols):
            return 1
        raise ValueError("Could not detect format (missing expected columns for either format 1 or 2).")

    @staticmethod
    def build_ingredient_from_format1_row(row: Dict[str, Any]) -> str:
        cond = clean_str(row.get("condition"))
        reagent = clean_str(row.get("reagent"))
        if reagent is None:
            return cond if cond is not None else ""
        if cond is None or cond == reagent:
            return reagent
        return f"{cond}::{reagent}"

    @staticmethod
    def choose_source_and_conc_from_format1_row(row: Dict[str, Any]) -> (Optional[str], Optional[float], Optional[str]):
        reagent = clean_str(row.get("reagent"))
        unit = unit_normalize(row.get("unit"))
        stock_x = to_float(row.get("stock_x"))
        stock_conc = to_float(row.get("stock_conc"))
        if reagent is None:
            return None, None, unit
        if unit == "1X_dilution":
            conc = stock_x if stock_x is not None else stock_conc
        else:
            conc = stock_conc if stock_conc is not None else stock_x
        return reagent, conc, unit

    @staticmethod
    def coerce_atoms(row: Dict[str, Any], col: str) -> int:
        v = to_int_maybe(row.get(col))
        if v is None:
            return 0
        if isinstance(v, float):
            return int(round(v))
        return int(v)

    @staticmethod
    def ingredient_id_acronym(ingredient: str) -> str:
        base = ingredient.split("::")[-1]
        tokens = re.split(r"[^A-Za-z0-9]+", base)
        tokens = [t for t in tokens if t]
        if not tokens:
            return "ING"
        acronym = "".join(t[0] for t in tokens if t)[:4].upper()
        if len(acronym) < 3:
            first = tokens[0].upper()
            need = 3 - len(acronym)
            acronym += first[1:1 + need] if len(first) > 1 else ("X" * need)
        return acronym

    @staticmethod
    def ensure_unique_id(base_id: str, used: set) -> str:
        if base_id not in used:
            used.add(base_id)
            return base_id
        i = 2
        while True:
            candidate = f"{base_id}{i}"
            if candidate not in used:
                used.add(candidate)
                return candidate
            i += 1

    @staticmethod
    def load_id_map(path: Optional[str]) -> Dict[str, str]:
        if not path:
            return {}
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        expected = {"ingredient", "id"}
        cols = set(df.columns)
        if not expected.issubset(cols):
            raise ValueError("ID map must contain columns INGREDIENT and ID.")
        mapping = {}
        for _, r in df.iterrows():
            ing = clean_str(r.get("ingredient"))
            mid = clean_str(r.get("id"))
            if ing and mid:
                mapping[ing] = mid
        return mapping

    @staticmethod
    def convert_format1(df: pd.DataFrame, id_map: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
        ingredients = []
        used_ids = set()
        for _, row in df.iterrows():
            r = {k: row.get(k) for k in df.columns}
            ingredient = IngredientsConfig.build_ingredient_from_format1_row(r)
            if not ingredient:
                continue
            base_id = id_map.get(ingredient)
            if not base_id:
                reagent = clean_str(r.get("reagent"))
                cond = clean_str(r.get("condition"))
                if reagent is None and cond:
                    base_id = cond
                else:
                    base_id = IngredientsConfig.ingredient_id_acronym(ingredient)
            base_id = IngredientsConfig.ensure_unique_id(base_id, used_ids)
            source, source_conc, unit = IngredientsConfig.choose_source_and_conc_from_format1_row(r)
            out = {
                "ID": base_id,
                "INGREDIENT": ingredient,
                "TYPE": clean_str(r.get("type")),
                "N_STATES": to_int_maybe(r.get("n_states")),
                "NOMINAL_VALUE": to_float(r.get("nominal_value")),
                "MIN_VALUE": to_float(r.get("min_value")),
                "MAX_VALUE": to_float(r.get("max_value")),
                "SOURCE": source,
                "SOURCE_CONC": source_conc,
                "UNIT": unit,
                "C_ATOMS": IngredientsConfig.coerce_atoms(r, "c_atoms"),
                "N_ATOMS": IngredientsConfig.coerce_atoms(r, "n_atoms"),
                "NA_ATOMS": IngredientsConfig.coerce_atoms(r, "na_atoms"),
                "CL_ATOMS": IngredientsConfig.coerce_atoms(r, "cl_atoms"),
            }
            ingredients.append(out)
        return {"ingredients": ingredients}

    @staticmethod
    def convert_format2(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        ingredients = []
        for _, row in df.iterrows():
            r = {k: row.get(k) for k in df.columns}
            out = {
                "ID": clean_str(r.get("id")),
                "INGREDIENT": clean_str(r.get("ingredient")),
                "TYPE": clean_str(r.get("type")),
                "N_STATES": to_int_maybe(r.get("n_states")),
                "NOMINAL_VALUE": to_float(r.get("nominal_value")),
                "MIN_VALUE": to_float(r.get("min_value")),
                "MAX_VALUE": to_float(r.get("max_value")),
                "SOURCE": clean_str(r.get("source")),
                "SOURCE_CONC": to_float(r.get("source_conc")),
                "UNIT": unit_normalize(r.get("unit")),
                "C_ATOMS": int(to_int_maybe(r.get("c_atoms")) or 0),
                "N_ATOMS": int(to_int_maybe(r.get("n_atoms")) or 0),
                "NA_ATOMS": int(to_int_maybe(r.get("na_atoms")) or 0),
                "CL_ATOMS": int(to_int_maybe(r.get("cl_atoms")) or 0),
            }
            ingredients.append(out)
        return {"ingredients": ingredients}

    @staticmethod
    def finalize_numeric_types(result: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        for ing in result.get("ingredients", []):
            ns = ing.get("N_STATES")
            if isinstance(ns, float) and abs(ns - round(ns)) < 1e-9:
                ing["N_STATES"] = int(round(ns))
            for fld in ["NOMINAL_VALUE", "MIN_VALUE", "MAX_VALUE", "SOURCE_CONC"]:
                v = ing.get(fld)
                if isinstance(v, float) and abs(v - round(v)) < 1e-9:
                    ing[fld] = float(v)
        return result

    @staticmethod
    def parse_ingredients(input_path: str, sheet: Optional[Union[str, int]], id_map_path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Read and convert the ingredients table (Format 1 or 2) into a dict {"ingredients": [...]}
        without performing any user-facing I/O.
        """
        df = read_table(input_path, sheet_name=sheet)
        if isinstance(df, dict):
            raise ValueError(f"Multiple sheets detected; please select one with -s. Available sheets: {list(df.keys())}")
        df = normalize_columns(df)

        fmt = IngredientsConfig.detect_format(df)
        if fmt == 2:
            required = ["id", "ingredient", "type", "nominal_value", "min_value", "max_value"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns for format 2: {missing}")
            result = IngredientsConfig.convert_format2(df)
        else:
            required = ["condition", "reagent", "type", "nominal_value", "min_value", "max_value"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns for format 1: {missing}")
            id_map = IngredientsConfig.load_id_map(id_map_path)
            result = IngredientsConfig.convert_format1(df, id_map=id_map)

        for ing in result.get("ingredients", []):
            for k, v in list(ing.items()):
                if isinstance(v, float) and pd.isna(v):
                    ing[k] = None

        result = IngredientsConfig.finalize_numeric_types(result)
        return result


def to_json_string(obj: Any) -> str:
    def nan_to_none(o):
        if isinstance(o, float) and pd.isna(o):
            return None
        return o
    return json.dumps(obj, indent=2, default=nan_to_none)