"""A TOML writer for capture.toml: bare keys, each value an array of strings.

X_eTaL's ``[]L_IST`` reads top-level arrays of strings only (docs/design.md, D2),
so that is all this writes. Output is ASCII: other characters are escaped.
"""

import re

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_SIMPLE = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _char(c: str) -> str:
    if c in _SIMPLE:
        return _SIMPLE[c]
    code = ord(c)
    if 0x20 <= code < 0x7F:
        return c
    return f"\\u{code:04X}" if code <= 0xFFFF else f"\\U{code:08X}"


def quote(s: str) -> str:
    return '"' + "".join(_char(c) for c in s) + '"'


def _line(key, values) -> str:
    if not isinstance(key, str) or not _BARE_KEY.fullmatch(key):
        raise ValueError(f"not a bare TOML key: {key!r}")
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError(f"{key}: value must be a list of strings")
    return f"{key} = [{', '.join(quote(v) for v in values)}]\n"


def dumps(table: dict) -> str:
    """The table as TOML text, keys in the given order."""
    return "".join(_line(k, v) for k, v in table.items())
