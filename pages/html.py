"""The site's HTML pages around the pictures X_eTaL drew (pages/build.py)."""

import re
from html import escape
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/softwarewrighter/X_eTaL-circuits"
OBSERVATION = (
    "Attention pictures and scores show where heads look: observations, not causal "
    "evidence. Which heads the output needs is tested by interventions, in later work."
)

CSS = """body { font-family: system-ui, sans-serif; max-width: 980px; margin: 0 auto;
  padding: 0 16px 48px; color: #1f2937; line-height: 1.5; }
nav { display: flex; gap: 18px; padding: 14px 0; border-bottom: 1px solid #e5e7eb; }
nav a { color: #1d4ed8; text-decoration: none; font-weight: 600; }
.about { font-size: 14px; color: #4b5563; }
.observation { background: #fef3c7; border-left: 4px solid #f59e0b; padding: 8px 12px;
  font-size: 14px; }
.picture { overflow-x: auto; }
.picture svg, .picture img { max-width: none; }
pre { background: #f3f4f6; padding: 12px; overflow-x: auto; font-size: 13px; }
select, button { font-size: 15px; margin-right: 8px; }
button { cursor: pointer; }
code { background: #f3f4f6; padding: 0 3px; }
"""


def prompt_words(meta: dict) -> str:
    half = (len(meta["token_ids"]) - 1) // 2
    return f"a beginning-of-text token, then {half} random tokens, then the same {half} again"


def page(title: str, meta: dict, body: str) -> str:
    """A whole page: the navigation, what is shown, the observation note, then body."""
    revision = meta["revision"][0]
    about = (
        f"{escape(meta['model'][0])} at revision <code>{escape(revision)}</code>, "
        f"{meta['dtype'][0]}, on {prompt_words(meta)}. Tokens: "
        + " | ".join(f"<code>{escape(t)}</code>" for t in meta["tokens"])
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} - X_eTaL-circuits</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<nav><a href="index.html">Model map</a><a href="heads.html">Heads</a>
<a href="scores.html">Scores</a><a href="effects.html">Effects</a>
<a href="eli5.html">Explain it simply</a>
<a href="{REPO}">Source</a></nav>
<h1>{escape(title)}</h1>
<p class="about">{about}</p>
<p class="observation">{OBSERVATION}</p>
{body}
</body>
</html>
"""


def index_body(out: Path, circuit: dict) -> str:
    svg = (out / "map.svg").read_text()
    return f"""<p>GPT-2 Small in one picture: a row per layer, a cell per attention head
colored by its scores (blue induction, orange previous token), an MLP cell
per layer, and the circuit <b>{escape(circuit["name"][0])}</b> drawn over it,
each edge dashed (observed), solid (tested) or thick (confirmed). Hover a
head for its numbers; see any head's attention on the <a href="heads.html">Heads</a>
page. Drawn by <a href="https://github.com/softwarewrighter/X_eTaL">X_eTaL</a>.</p>
<div class="picture">
{svg}</div>
"""


def options(values: list[int], chosen: int) -> str:
    return "".join(
        f'<option value="{v}"{" selected" if v == chosen else ""}>{v}</option>' for v in values
    )


HEADS_JS = """<script>
const layer = document.getElementById("layer"), head = document.getElementById("head");
const pic = document.getElementById("pic"), name = document.getElementById("name");
function show() {
  pic.src = "heads/L" + layer.value + "H" + head.value + ".svg";
  name.textContent = "head " + layer.value + "." + head.value;
  history.replaceState(null, "", "#" + layer.value + "." + head.value);
}
function pick(text) {
  const [l, h] = text.split(".");
  if ([...layer.options].some(o => o.value === l)) layer.value = l;
  if ([...head.options].some(o => o.value === h)) head.value = h;
  show();
}
layer.onchange = show; head.onchange = show;
document.querySelectorAll("button[data-head]").forEach(b => b.onclick = () => pick(b.dataset.head));
if (location.hash.length > 1) pick(location.hash.slice(1));
</script>
"""


def heads_body(circuit: dict, heads: list[tuple[int, int]]) -> str:
    first = (5, 1) if (5, 1) in heads else heads[0]
    layers = sorted({layer for layer, _ in heads})
    hs = sorted({h for _, h in heads})
    buttons = " ".join(
        f'<button data-head="{h}">{h} ({escape(r)})</button>'
        for h, r in zip(circuit["heads"], circuit["roles"], strict=True)
    )
    links = " ".join(f'<a href="heads/L{lay}H{h}.svg">{lay}.{h}</a>' for lay, h in heads)
    return f"""<p>One head's attention: a row per query token, a column per key token,
each cell the weight in percent. Heads are named layer.head, from 0.</p>
<p><label>Layer <select id="layer">{options(layers, first[0])}</select></label>
<label>Head <select id="head">{options(hs, first[1])}</select></label>
<b id="name">head {first[0]}.{first[1]}</b></p>
<p>The circuit's heads: {buttons}</p>
<div class="picture"><img id="pic" src="heads/L{first[0]}H{first[1]}.svg"
alt="attention heatmap of one head"></div>
<details><summary>Every head as a link</summary><p>{links}</p></details>
{HEADS_JS}"""


def scores_body(report: str) -> str:
    return f"""<p>Every head's previous-token score (its mean weight on the position
before) and induction score (its mean weight, on the second copy, on the token
after the current token's earlier occurrence). The leading heads:</p>
<pre>{escape(report)}</pre>
<div class="picture"><img src="scores/induction.svg" alt="induction score of every head"></div>
<div class="picture"><img src="scores/previous.svg" alt="previous-token score of every head"></div>
"""


def effects_body(reports: dict) -> str:
    grids = "".join(
        f"<h2>{escape(title)}</h2>\n<pre>{escape(reports[kind])}</pre>\n"
        + "".join(
            f'<div class="picture"><img src="effects/{m}.svg" alt="{m}: every head"></div>\n'
            for m in methods
        )
        for kind, title, methods in (
            (
                "patch",
                "Patching: copying one head's output between the twins",
                ("denoise", "noise"),
            ),
            ("ablation", "Ablation: switching one head off", ("mean", "zero")),
        )
    )
    confirm = (
        "<h2>Confirmed on held-out seeds</h2>\n"
        "<p>The heads that pass on the finding seeds, run again on ten held-out seeds with "
        "the same rule: <b>confirmed</b> when they pass again.</p>\n"
        f"<pre>{escape(reports['confirm'])}</pre>\n"
        '<div class="picture"><img src="effects/confirmed-denoise.svg" alt="confirmed heads, '
        'denoising"></div>\n'
        '<div class="picture"><img src="effects/confirmed-mean.svg" alt="confirmed heads, '
        'mean ablation"></div>\n'
    )
    verdict = (
        "<h2>Which heads beat their random controls</h2>\n"
        "<p>A head passes when its mean effect is at least 5% and larger than every one of "
        "20 random changes of the same size: <b>tested</b> on the finding seeds.</p>\n"
        f"<pre>{escape(reports['verdict'])}</pre>\n"
        '<div class="picture"><img src="effects/passing-denoise.svg" alt="passing heads, '
        'denoising"></div>\n'
        '<div class="picture"><img src="effects/passing-mean.svg" alt="passing heads, '
        'mean ablation"></div>\n'
    )
    return (
        confirm
        + verdict
        + f"""<p>Interventions on every head. Each repeated prompt has a twin whose
second copy is fresh tokens; the metric is the logit difference between the
right next token and the twin's, at each position of the second copy. Effects
are in percent, the mean over 10 seeds (blue below 0, red above). The grids
below show every head; the verdicts above say which beat their controls.</p>
{grids}"""
    )


def eli5_body() -> str:
    text = (ROOT / "docs" / "eli5.md").read_text()
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    return re.sub(r'href="([\w./-]+\.md)"', rf'href="{REPO}/blob/main/docs/\1"', body)


def write(out: Path, meta: dict, circuit: dict, heads: list, report: str, effects: dict) -> None:
    (out / "style.css").write_text(CSS)
    pages = {
        "index.html": ("GPT-2 Small: the model map", index_body(out, circuit)),
        "heads.html": ("GPT-2 Small: one head's attention", heads_body(circuit, heads)),
        "scores.html": ("GPT-2 Small: every head's scores", scores_body(report)),
        "effects.html": ("GPT-2 Small: interventions on every head", effects_body(effects)),
        "eli5.html": ("Explain it simply", eli5_body()),
    }
    for name, (title, body) in pages.items():
        (out / name).write_text(page(title, meta, body))
