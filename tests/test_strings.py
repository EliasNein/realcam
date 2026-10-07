import ast
import re
import string
from pathlib import Path

from gui import strings

GUI = Path(strings.__file__).parent
KEY = re.compile(r"^[a-z_]+(\.[a-z0-9_]+)+$")


def _constants():
    """Every string constant in the GUI sources except strings.py itself."""
    found = set()
    for f in GUI.rglob("*.py"):
        if f.name == "strings.py":
            continue
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(node.value)
    return found


def _namespaces():
    return {k.split(".")[0] for lang in strings.STRINGS.values() for k in lang}


def test_languages_have_identical_keys():
    langs = list(strings.STRINGS.values())
    for other in langs[1:]:
        assert set(other) == set(langs[0])


def test_placeholders_match_between_languages():
    langs = list(strings.STRINGS.values())

    def fields(text):
        return {name for _, name, _, _ in string.Formatter().parse(text) if name}

    for key in langs[0]:
        for other in langs[1:]:
            assert fields(other[key]) == fields(langs[0][key]), key


def test_every_key_used_in_code_exists():
    ns = _namespaces()
    used = {c for c in _constants() if KEY.match(c) and c.split(".")[0] in ns}
    missing = {k for k in used if any(k not in lang for lang in strings.STRINGS.values())}
    assert not missing


def test_no_unused_keys():
    used = _constants()
    unused = [k for k in strings.STRINGS["de"] if k not in used]
    assert not unused


def test_t_fills_placeholders():
    assert "5" in strings.t("preflight.vram.low", vram="3", recommended=5)
