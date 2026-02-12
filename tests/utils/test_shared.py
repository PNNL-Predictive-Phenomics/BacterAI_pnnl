"""Tests for configuration shared helpers."""
import pytest
import pandas as pd

from src.bacterai.configuration.shared import (
    is_nullish,
    clean_str,
    normalize_key,
    parse_bool,
    parse_int,
    parse_float,
    parse_list,
    unit_normalize,
    normalize_columns,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, True),
        ("", True),
        ("NA", True),
        ("n/a", True),
        ("  ", True),
        (0, False),
        ("ok", False),
    ],
)
def test_is_nullish(value, expected):
    assert is_nullish(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("  hello ", "hello"),
        ("NA", None),
        (None, None),
        ("", None),
    ],
)
def test_clean_str(value, expected):
    assert clean_str(value) == expected


def test_normalize_key():
    assert normalize_key(" My Key ") == "my_key"
    assert normalize_key("A  B") == "a_b"
    assert normalize_key("A-B") == "ab"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("Y", True),
        ("0", False),
        ("no", False),
        ("", None),
    ],
)
def test_parse_bool(value, expected):
    assert parse_bool(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("10", 10),
        ("10.0", 10),
        ("-2", -2),
        ("", None),
    ],
)
def test_parse_int(value, expected):
    assert parse_int(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("10.5", 10.5),
        ("-2", -2.0),
        ("", None),
    ],
)
def test_parse_float(value, expected):
    assert parse_float(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1,2,3", [1, 2, 3]),
        ("[1,2]", [1, 2]),
        ("", None),
    ],
)
def test_parse_list_int(value, expected):
    assert parse_list(value, "int") == expected


def test_unit_normalize():
    assert unit_normalize("x") == "1X_dilution"
    assert unit_normalize("mM") == "mM"
    assert unit_normalize("") is None


def test_normalize_columns():
    df = pd.DataFrame({"A Col": [1], "B  Col": [2]})
    out = normalize_columns(df)
    assert list(out.columns) == ["a_col", "b_col"]
