"""Build the static site from committed captures (scripts/pages.sh, `just pages`).

Every picture is drawn by X_eTaL through the command-line scripts, exactly as
at the command line; the pages only show and pick among them. Same inputs,
same bytes: nothing here reads the clock or the git state.

    adapter/.venv/bin/python -m pages.build OUT_DIR
"""

import shutil
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pages import html

ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "fixtures" / "gpt2-induction-full"
CIRCUIT = ROOT / "experiments" / "circuits" / "gpt2-induction.toml"


def script(name: str, *args: str) -> str:
    """Run one of scripts/ and give its standard output."""
    r = subprocess.run(
        [str(ROOT / "scripts" / name), *args], cwd=ROOT, capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"scripts/{name} {' '.join(args)}: {r.stderr}")
    return r.stdout


def all_heads(capture: Path) -> list[tuple[int, int]]:
    meta = tomllib.loads((capture / "capture.toml").read_text())
    heads = int(meta["attn_shape"][1])
    return [(int(layer), h) for layer in meta["attn_layers"] for h in range(heads)]


def draw_heads(out: Path, heads: list[tuple[int, int]]) -> None:
    (out / "heads").mkdir()

    def one(lh: tuple[int, int]) -> None:
        layer, head = lh
        svg = script("heatmap.sh", str(CAPTURE), str(layer), str(head))
        (out / "heads" / f"L{layer}H{head}.svg").write_text(svg)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, heads))


def build(out: Path, heads: list[tuple[int, int]] | None = None) -> None:
    """Build the site into out (emptied first); heads defaults to every head."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    heads = all_heads(CAPTURE) if heads is None else heads
    draw_heads(out, heads)
    report = script("scores.sh", str(CAPTURE), str(out / "scores"))
    script("map.sh", str(CAPTURE), str(CIRCUIT), str(out / "map.svg"))
    effects = {
        kind: script(
            "effects.sh", str(ROOT / "fixtures" / f"gpt2-induction-{kind}"), str(out / "effects")
        )
        for kind in ("patch", "ablation")
    }
    meta = tomllib.loads((CAPTURE / "capture.toml").read_text())
    circuit = tomllib.loads(CIRCUIT.read_text())
    html.write(out, meta, circuit, heads, report, effects)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m pages.build OUT_DIR")
    build(Path(sys.argv[1]))
