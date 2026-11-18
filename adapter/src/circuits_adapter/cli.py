"""Command-line entry point for the adapter."""

import argparse

from circuits_adapter import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="circuits-adapter",
        description="Run a model and export its internals for X_eTaL-circuits.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
