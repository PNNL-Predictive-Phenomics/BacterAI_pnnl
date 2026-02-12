"""Tests for IngredientsConfig parsing and conversion."""
import json
import pandas as pd
import pytest

from src.bacterai.configuration.ingredients import IngredientsConfig


def _write_csv(path, df):
    df.to_csv(path, index=False)


def test_detect_format_errors():
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError):
        IngredientsConfig.detect_format(df)


def test_parse_format1_with_id_map(temp_experiment_dir):
    csv_path = temp_experiment_dir / "ingredients_f1.csv"
    id_map_path = temp_experiment_dir / "id_map.csv"

    df = pd.DataFrame({
        "condition": ["base", "base"],
        "reagent": ["glucose", "ammonia"],
        "type": ["quantitative", "binary"],
        "nominal_value": [1, 0],
        "min_value": [0, 0],
        "max_value": [10, 1],
        "n_states": [10, 2],
        "unit": ["mM", "mM"],
        "stock_x": [1, 1],
    })
    _write_csv(csv_path, df)

    id_map = pd.DataFrame({"ingredient": ["base::glucose"], "id": ["GLU"]})
    _write_csv(id_map_path, id_map)

    result = IngredientsConfig.parse_ingredients(
        input_path=str(csv_path),
        sheet=None,
        id_map_path=str(id_map_path),
    )

    ingredients = result["ingredients"]
    names = [ing["INGREDIENT"] for ing in ingredients]
    ids = [ing["ID"] for ing in ingredients]

    assert "base::glucose" in names
    assert "base::ammonia" in names
    assert "GLU" in ids


def test_parse_format2_basic(temp_experiment_dir):
    csv_path = temp_experiment_dir / "ingredients_f2.csv"
    df = pd.DataFrame({
        "id": ["GLU", "NH4"],
        "ingredient": ["glucose", "ammonia"],
        "type": ["quantitative", "binary"],
        "nominal_value": [1, 0],
        "min_value": [0, 0],
        "max_value": [10, 1],
        "n_states": [10, 2],
        "source": ["glucose", "ammonia"],
        "source_conc": [1.0, 1.0],
        "unit": ["mM", "mM"],
        "c_atoms": [6, 0],
        "n_atoms": [0, 1],
        "na_atoms": [0, 0],
        "cl_atoms": [0, 0],
    })
    _write_csv(csv_path, df)

    result = IngredientsConfig.parse_ingredients(
        input_path=str(csv_path),
        sheet=None,
        id_map_path=None,
    )
    ingredients = result["ingredients"]
    assert ingredients[0]["ID"] == "GLU"
    assert ingredients[1]["INGREDIENT"] == "ammonia"


def test_load_id_map_requires_columns(temp_experiment_dir):
    bad_path = temp_experiment_dir / "bad_id_map.csv"
    pd.DataFrame({"foo": ["a"], "bar": ["b"]}).to_csv(bad_path, index=False)

    with pytest.raises(ValueError):
        IngredientsConfig.load_id_map(str(bad_path))
