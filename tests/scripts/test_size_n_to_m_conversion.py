"""Tests for ingredient size conversion utilities."""
import pandas as pd

from src.bacterai.scripts.size_n_to_m_conversion import fill_new_ingredients


def test_fill_new_ingredients_right():
    data = pd.DataFrame({"a": [1], "b": [2], "y_true": [0.1]})
    out = fill_new_ingredients(data, original_size=2, fill_column_names=["c"], fill_on_right=True)

    assert list(out.columns) == ["a", "b", "c", "y_true"]
    assert out.loc[0, "c"] == 1


def test_fill_new_ingredients_left():
    data = pd.DataFrame({"a": [1], "b": [2]})
    out = fill_new_ingredients(data, original_size=2, fill_column_names=["c"], fill_on_right=False)

    assert list(out.columns) == ["c", "a", "b"]
    assert out.loc[0, "c"] == 1
