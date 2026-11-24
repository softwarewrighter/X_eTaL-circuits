"""Command-line entry point for the adapter."""

import argparse

from circuits_adapter import __version__


def _layers(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _capture_parser(sub) -> None:
    p = sub.add_parser("capture", help="run a prompt and write a capture directory")
    p.add_argument("--out", required=True, help="new directory for the capture")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--revision", default=None, help="model revision (pinned default for gpt2)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    prompt = p.add_mutually_exclusive_group()
    prompt.add_argument("--text", default=None, help="a text prompt (BOS is prepended)")
    prompt.add_argument(
        "--length", type=int, default=50, help="repeated random tokens: length of one copy"
    )
    p.add_argument("--layers", type=_layers, default=None, help="comma list, default all")
    p.add_argument("--safetensors", action="store_true", help="also write capture.safetensors")


def _baseline_parser(sub) -> None:
    p = sub.add_parser("baseline", help="the experiment's metric on a clean prompt and its twin")
    p.add_argument("--experiment", required=True, help="an experiments/*.toml file")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True, help="new directory for the run")
    p.add_argument("--device", default="cpu")


def _ablate_parser(sub) -> None:
    p = sub.add_parser("ablate", help="each head zero- and mean-ablated: the metric's drop")
    p.add_argument("--experiment", required=True, help="an experiments/*.toml file")
    p.add_argument(
        "--seeds", default="finding", help="finding, heldout, or a comma list (default finding)"
    )
    p.add_argument("--out", required=True, help="new directory: one run per seed, seed-N")
    p.add_argument("--device", default="cpu")


def _patch_parser(sub) -> None:
    p = sub.add_parser("patch", help="activation patching between the twins, per seed")
    p.add_argument("--experiment", required=True, help="an experiments/*.toml file")
    p.add_argument(
        "--seeds", default="finding", help="finding, heldout, or a comma list (default finding)"
    )
    p.add_argument("--out", required=True, help="new directory: one run per seed, seed-N")
    p.add_argument("--device", default="cpu")


def seeds_of(exp, text: str) -> tuple[int, ...]:
    """The seeds named on the command line: finding, heldout or a comma list."""
    named = {"finding": exp.finding_seeds, "heldout": exp.heldout_seeds}
    return named[text] if text in named else tuple(_layers(text))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="circuits-adapter",
        description="Run a model and export its internals for X_eTaL-circuits.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")
    _capture_parser(sub)
    _baseline_parser(sub)
    _ablate_parser(sub)
    _patch_parser(sub)
    return parser


def run_capture(args) -> int:
    from circuits_adapter import capture, model, prompts

    m = model.load(args.model, args.revision, args.device)
    if args.text is not None:
        ids = model.encode(m, args.text)
    else:
        ids = prompts.repeated_random(args.seed, args.length, m.bos)
    c = model.capture(m, ids, args.seed, args.layers, prompt=args.text or "")
    print(capture.write(args.out, c, safetensors=args.safetensors))
    return 0


def run_baseline(args) -> int:
    from circuits_adapter import baseline, experiment, model

    exp = experiment.load(args.experiment)
    m = model.load(exp.model, exp.revision, args.device)
    print(baseline.write(args.out, m, exp, args.seed))
    return 0


def run_ablate(args) -> int:
    from pathlib import Path

    from circuits_adapter import ablate, experiment, model

    exp = experiment.load(args.experiment)
    m = model.load(exp.model, exp.revision, args.device)
    means = ablate.twin_means(m, exp)
    for seed in seeds_of(exp, args.seeds):
        print(ablate.write(Path(args.out) / f"seed-{seed}", m, exp, seed, means))
    return 0


def run_patch(args) -> int:
    from pathlib import Path

    from circuits_adapter import experiment, model, patch

    exp = experiment.load(args.experiment)
    m = model.load(exp.model, exp.revision, args.device)
    for seed in seeds_of(exp, args.seeds):
        print(patch.write(Path(args.out) / f"seed-{seed}", m, exp, seed))
    return 0


COMMANDS = {
    "capture": run_capture,
    "baseline": run_baseline,
    "ablate": run_ablate,
    "patch": run_patch,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in COMMANDS:
        return COMMANDS[args.command](args)
    parser.print_help()
    return 2
