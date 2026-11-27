"""The site's introduction, glossary, findings and reading notes (for a first-time reader)."""

from html import escape
from pathlib import Path

import markdown

CONTENT = Path(__file__).resolve().parent / "content"
G = "glossary.html#"

GUIDE = (
    ("map.html", "The model map", "the whole model in one picture: every layer, head and MLP, "
     "heads colored by where they look, the circuit drawn over it with its evidence"),
    ("heads.html", "Heads", "any one head's attention as a heatmap: which earlier word each "
     "word looks at"),
    ("scores.html", "Scores", "every head ranked by how much it looks like an induction or a "
     "previous-token head (observation only)"),
    ("effects.html", "Effects", "what switching each head off or patching it does, which heads "
     "beat random controls, and which hold up on held-out prompts"),
    ("edges.html", "Edges", "which head feeds which: path patching of the strongest edges "
     "against random controls"),
    ("glossary.html", "Glossary", "every term these pages use"),
    ("eli5.html", "Explain it simply", "the paper and this project, explained from scratch"),
)  # fmt: skip

HOW_TO_READ = {
    "map.html": f"""One row per <a href="{G}layer">layer</a>, the first at the bottom; one
cell per <a href="{G}attention-head">head</a> (named layer.head) and one per
<a href="{G}mlp">MLP</a>. Blue cells look like <a href="{G}induction-head">induction
heads</a>, orange like <a href="{G}previous-token-head">previous-token heads</a>,
deeper for higher; gray neither. Outlined heads are part of the circuit, outlined by
their <a href="{G}evidence-level">evidence</a>: dashed observed, solid tested, thick
confirmed. Red arrows are <a href="{G}edge">edges</a>, styled the same way. Hover a
head for its numbers.""",
    "heads.html": f"""One head's <a href="{G}attention">attention</a>: a row per word that
looks (top to bottom, in order), a column per word looked at, each cell the percent of
that row's attention. An induction head shows a bright stripe just below the diagonal
in the second half: each word looks at the word after its earlier copy.""",
    "scores.html": f"""Each head's average attention to the previous word
(<a href="{G}previous-token-head">previous-token score</a>) and to the word after the
earlier copy (<a href="{G}induction-head">induction score</a>), in percent. These name
suspects: they say where heads look, not whether the model needs them.""",
    "effects.html": f"""What happens to the <a href="{G}logit-difference">logit
difference</a> when a head is switched off (<a href="{G}ablation">ablation</a>) or its
output is copied between the <a href="{G}twin">twins</a> (<a href="{G}patching">patching</a>),
in percent: red helps, blue hurts. Verdicts compare each head with 20 random
<a href="{G}control">controls</a>, then check it on <a href="{G}held-out-seeds">held-out
seeds</a>.""",
    "edges.html": f"""Each grid shows <a href="{G}edge">edges</a> that beat their
random <a href="{G}control">controls</a>, first those confirmed on
<a href="{G}held-out-seeds">held-out seeds</a>: a row per sending head, a column per receiving
head, each cell the percent of the receiver's gap that the sender's output restores when
it reaches that receiver alone (<a href="{G}patching">path patching</a>). Red helps,
blue pushes away; 0 means the edge was not tested or did not pass.""",
    "glossary.html": "Terms in alphabetical order; every page links here.",
    "eli5.html": "The long introduction: the paper, the project, and why looking is not proving.",
    "index.html": "",
}


def render(name: str) -> str:
    """A content file's markdown as HTML, headings given ids (glossary.html#attention-head)."""
    text = (CONTENT / name).read_text()
    return markdown.markdown(text, extensions=["toc"])


def findings(circuit: dict) -> str:
    """What the runs found so far, straight from the circuit file: a row per head."""
    rows = "".join(
        f"<tr><td>{escape(h)}</td><td>{escape(r)}</td><td><b>{escape(e)}</b></td>"
        f"<td>{escape(b)}</td></tr>\n"
        for h, r, e, b in zip(
            circuit["heads"],
            circuit["roles"],
            circuit["head_evidence"],
            circuit["head_basis"],
            strict=True,
        )
    )
    confirmed = sum(e == "confirmed" for e in circuit["head_evidence"])
    edges = sorted(set(circuit["evidence"]))
    return f"""<h2>What we found so far</h2>
<p>GPT-2 Small's copying needs or uses {confirmed} heads that pass every check
(<b>confirmed</b>): the previous-token head, the five induction heads, and six more
that the interventions found and whose roles are not yet known. Which head feeds which
is tested edge by edge on the <a href="edges.html">Edges</a> page: 4.11 feeds every
induction head's key, and the induction heads feed the queries of 7.6, 9.6, 9.9 and
10.1, against random controls and again on held-out prompts. The circuit's
{len(circuit["edges"])} arrows are <b>{", ".join(edges)}</b>.</p>
<table>
<tr><th>Head</th><th>Role</th><th>Evidence</th><th>Basis (percent effects)</th></tr>
{rows}</table>
"""


def guide() -> str:
    items = "".join(
        f'<li><a href="{href}">{escape(title)}</a>: {escape(what)}</li>\n'
        for href, title, what in GUIDE
    )
    return f"<h2>The pages</h2>\n<ul>\n{items}</ul>\n"


def index_body(circuit: dict) -> str:
    return (
        render("intro.md")
        + findings(circuit)
        + guide()
        + '<p><a href="map.html"><img src="map.svg" alt="the model map" width="640"></a></p>\n'
    )


def how_to_read(page: str) -> str:
    text = HOW_TO_READ[page]
    if not text:
        return ""
    return f'<p class="how"><b>How to read this page.</b> {text}</p>\n'
