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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="circuits-adapter",
        description="Run a model and export its internals for X_eTaL-circuits.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _capture_parser(parser.add_subparsers(dest="command"))
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "capture":
        return run_capture(args)
    parser.print_help()
    return 2
