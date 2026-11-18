"""capture.toml: every value an array of strings (docs/design.md, D2)."""

import tomllib

import pytest

from circuits_adapter import toml_out


def test_values_are_arrays_of_strings_and_round_trip():
    meta = {"model": ["gpt2"], "tokens": [" the", "\n", 'say "hi"', "back\\slash", "é\x01"]}
    text = toml_out.dumps(meta)
    assert tomllib.loads(text) == meta


def test_key_order_is_kept():
    text = toml_out.dumps({"b": ["1"], "a": ["2"]})
    assert text.index("b =") < text.index("a =")


def test_output_is_ascii():
    text = toml_out.dumps({"tokens": ["é", "\U0001f600"]})
    assert text.isascii()
    assert tomllib.loads(text)["tokens"] == ["é", "\U0001f600"]


@pytest.mark.parametrize("bad", [{"x": "not a list"}, {"x": [1]}, {"bad key": ["v"]}])
def test_refuses_non_string_lists_and_bare_key_violations(bad):
    with pytest.raises(ValueError):
        toml_out.dumps(bad)
