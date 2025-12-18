"""
Ingredients configuration processing.
"""
import re
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .shared import (
    clean_str,
    is_nullish,
    normalize_columns,
    read_table,
    to_float,
    to_int_maybe,
    unit_normalize,
)


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
