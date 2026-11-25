"""The static site (pages/build.py, `just pages`): built from committed captures by X_eTaL.

Most tests build a few heads only, to stay fast; the full build draws all 144.
"""

import hashlib
import re
import tomllib
from html.parser import HTMLParser
from pathlib import Path

import pytest

from pages import build

ROOT = Path(__file__).resolve().parents[2]
SOME = [(5, 1), (4, 11), (0, 0)]
SITE_URL = "https://softwarewrighter.github.io/X_eTaL-circuits/"

pytestmark = pytest.mark.skipif(
    not (ROOT / "bin" / "xetal").exists(), reason="bin/xetal not built (just xetal)"
)


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("site")
    build.build(out, heads=SOME)
    return out


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ("href", "src", "data") and v:
                self.refs.append(v)


def refs(page: Path) -> list[str]:
    p = Links()
    p.feed(page.read_text())
    return p.refs


def test_the_pages_and_pictures_are_there(site):
    for name in (
        "index.html",
        "heads.html",
        "scores.html",
        "effects.html",
        "eli5.html",
        "style.css",
    ):
        assert (site / name).is_file(), name
    assert sorted(p.name for p in (site / "heads").iterdir()) == sorted(
        f"L{layer}H{h}.svg" for layer, h in SOME
    )
    assert (site / "scores" / "induction.svg").is_file()
    assert (site / "scores" / "previous.svg").is_file()
    assert (site / "map.svg").is_file()
    assert sorted(p.name for p in (site / "effects").iterdir()) == [
        "denoise.svg",
        "mean.svg",
        "noise.svg",
        "passing-denoise.svg",
        "passing-mean.svg",
        "zero.svg",
    ]


def test_the_effects_page_shows_both_reports_and_says_candidates(site):
    text = (site / "effects.html").read_text()
    assert "denoising: percent of the clean-twin gap restored" in text
    assert "mean ablation: percent of the clean logit difference lost" in text
    assert "beat all 20 of their random controls" in text


def test_every_local_link_resolves_and_external_ones_are_https(site):
    for page in site.glob("*.html"):
        for ref in refs(page):
            if ref.startswith(("http://", "https://")):
                assert ref.startswith("https://"), (page.name, ref)
            elif not ref.startswith("#"):
                assert (site / ref.split("#")[0]).exists(), (page.name, ref)


def test_every_page_says_what_it_shows_and_that_it_is_an_observation(site):
    meta = tomllib.loads((ROOT / "fixtures/gpt2-induction-full/capture.toml").read_text())
    for page in site.glob("*.html"):
        text = page.read_text()
        assert "gpt2" in text and meta["revision"][0] in text, page.name
        assert "not causal evidence" in text, page.name
        assert "16 random tokens" in text, page.name


def test_the_pictures_are_x_etal_s(site):
    from subprocess import run

    want = run(
        [str(ROOT / "scripts/heatmap.sh"), "fixtures/gpt2-induction-full", "5", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert (site / "heads" / "L5H1.svg").read_text() == want
    assert (site / "map.svg").read_text() == (ROOT / "images/gpt2-model-map.svg").read_text()


def test_the_map_is_inline_on_the_front_page_so_hovering_works(site):
    index = (site / "index.html").read_text()
    assert '<g class="head" id="head-5.5">' in index


def test_the_head_picker_offers_the_built_heads_and_the_circuit(site):
    text = (site / "heads.html").read_text()
    assert re.search(r'<select id="layer"', text) and re.search(r'<select id="head"', text)
    for layer, h in SOME:
        assert f'href="heads/L{layer}H{h}.svg"' in text
    assert 'data-head="4.11"' in text and 'data-head="7.10"' in text


def test_the_eli5_page_is_the_doc_with_links_to_github(site):
    text = (site / "eli5.html").read_text()
    assert "Explain it like I" in text
    assert 'href="https://github.com/softwarewrighter/X_eTaL-circuits/blob/main/docs/plan.md"' in (
        text
    )


def test_same_inputs_same_bytes(site, tmp_path):
    build.build(tmp_path, heads=SOME)

    def files(d: Path) -> dict[str, bytes]:
        return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}

    assert files(tmp_path) == files(site)


def test_the_full_build_draws_every_head():
    assert build.all_heads(ROOT / "fixtures/gpt2-induction-full") == [
        (layer, h) for layer in range(12) for h in range(12)
    ]


def test_readme_links_the_live_demo_with_a_current_screenshot():
    readme = (ROOT / "README.md").read_text()
    assert f'<a href="{SITE_URL}">' in readme
    assert f'<a href="{SITE_URL}"><img src="images/demo.png"' in readme
    assert "No live site yet" not in readme
    source = (ROOT / "images/demo.png.source").read_text().split()
    digest = hashlib.sha256((ROOT / "images/gpt2-model-map.svg").read_bytes()).hexdigest()
    assert source == [digest, "images/gpt2-model-map.svg"], "retake it: just screenshot"


def test_the_workflow_builds_and_deploys_the_site():
    wf = (ROOT / ".github/workflows/pages.yml").read_text()
    for needed in (
        "branches: [main]",
        "scripts/xetal.sh",
        "scripts/pages.sh",
        "actions/upload-pages-artifact",
        "actions/deploy-pages",
    ):
        assert needed in wf, needed
