"""The Capture library (xetal/Capture/src/Capture.xtl) on the fixture and on broken copies.

Runs X_eTaL through scripts/xt (bin/xetal at XETAL_COMMIT, X_eTaL-ML at
XETAL_ML_COMMIT); skipped when bin/xetal is not built (`just xetal`).
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from circuits_adapter import xta

ROOT = Path(__file__).resolve().parents[3]
XT = ROOT / "scripts" / "xt"
FIXTURE = ROOT / "fixtures" / "gpt2-induction"
SRC = ROOT / "xetal" / "Capture" / "src" / "Capture.xtl"

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built (just xetal)"
)


def run(body: str) -> subprocess.CompletedProcess:
    src = '"cp:" u_se< "Capture"\n' + body
    return subprocess.run([str(XT), "eval", "-e", src], cwd=ROOT, capture_output=True, text=True)


def out(body: str) -> str:
    r = run(body)
    assert r.returncode == 0, r.stderr
    return r.stdout


def err(body: str) -> str:
    r = run(body)
    assert r.returncode != 0, r.stdout
    return r.stderr


def attn() -> np.ndarray:
    return xta.read(FIXTURE / "attn.xta").astype(np.float64)


D = f'"{FIXTURE}"'


def test_metadata_strings_and_integers():
    assert out(f'd_isclose f_irst {D} cp:t_exts "model"').strip() == "gpt2"
    assert out(f'{D} cp:i_nts "attn_shape"').split() == ["2", "12", "17", "17"]
    assert out(f'{D} cp:i_nts "attn_layers"').split() == ["0", "5"]


def test_tokens_one_per_position():
    assert out(f"t_ally cp:t_okens {D}").strip() == "17"


def test_read_gives_the_binary_files_values():
    printed = np.array(out(f'r_avel {D} cp:r_ead "attn"').split(), dtype=np.float64)
    assert np.array_equal(printed, attn().ravel())
    assert out(f'{D} cp:r_ead "logits"').split()[:1] != []


def test_attention_is_checked_and_returned():
    assert out(f"s_hape cp:a_ttention {D}").split() == ["2", "12", "17", "17"]


def test_selection_by_model_layer_head_and_query():
    assert out(f"{D} cp:p_osition 5").strip() == "2"
    head = np.array(out(f"r_avel (2, 6) cp:h_ead cp:a_ttention {D}").split(), dtype=float)
    assert np.array_equal(head, attn()[1, 5].ravel())
    row = np.array(out(f"(2, 6, 10) cp:q_uery cp:a_ttention {D}").split(), dtype=float)
    assert np.array_equal(row, attn()[1, 5, 9])
    assert out(f"s_hape 2 cp:l_ayer cp:a_ttention {D}").split() == ["12", "17", "17"]


def test_a_layer_not_captured_is_an_error():
    assert "error[capture]" in err(f"{D} cp:p_osition 7")


def test_a_relative_directory_is_refused():
    # X_eTaL would look for it beside the library, not in the working directory (QD9).
    assert "absolute path" in err('cp:a_ttention "fixtures/gpt2-induction"')


def broken(tmp_path: Path, edit) -> str:
    d = tmp_path / "cap"
    shutil.copytree(FIXTURE, d)
    edit(d)
    return f'cp:a_ttention "{d}"'


def edit_attn(d: Path, change) -> None:
    rows = [line.split() for line in (d / "attn.txt").read_text().splitlines()]
    change(rows)
    (d / "attn.txt").write_text("".join(" ".join(r) + "\n" for r in rows))


def test_refuses_a_wrong_format(tmp_path):
    def edit(d):
        t = (d / "capture.toml").read_text().replace("capture-1", "capture-9")
        (d / "capture.toml").write_text(t)

    assert "error[capture]" in err(broken(tmp_path, edit))


def test_refuses_a_count_that_does_not_fit_the_shape(tmp_path):
    assert "error[capture]" in err(broken(tmp_path, lambda d: edit_attn(d, lambda r: r[-1].pop())))


def test_refuses_attention_to_a_later_key(tmp_path):
    def change(rows):
        rows[1][5] = "0.25"  # layer 1, head 1, query 2 attends to key 6

    assert "not causal" in err(broken(tmp_path, lambda d: edit_attn(d, change)))


def test_refuses_rows_that_do_not_sum_to_one(tmp_path):
    def change(rows):
        rows[3][0] = repr(float(rows[3][0]) + 0.01)

    assert "sum to 1" in err(broken(tmp_path, lambda d: edit_attn(d, change)))


def test_doc_examples_pass():
    r = subprocess.run(
        [str(XT), "doc", "--test", str(SRC)], cwd=ROOT, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_export_types_are_pinned():
    r = subprocess.run([str(XT), "type", str(SRC)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout == (Path(__file__).parent / "types.out").read_text()


def test_lint_is_clean():
    r = subprocess.run([str(XT), "lint", str(SRC)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stderr == "", r.stderr
