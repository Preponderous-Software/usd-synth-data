"""``usd-synth`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import ConfigError, load_config


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.resolved:
        print(json.dumps(cfg.resolved(), indent=2, sort_keys=True))
    print(f"config OK: {args.config} sha256={cfg.config_hash()}")
    return 0


def _not_yet(milestone: str):
    def run(args: argparse.Namespace) -> int:
        print(f"usd-synth {args.command}: not implemented yet ({milestone})", file=sys.stderr)
        return 2

    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="usd-synth",
        description="Procedural synthetic-dataset generator (OpenUSD + Blender Cycles + COCO).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("config", help="validate a config and print its hash")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--resolved", action="store_true", help="also print the resolved config")
    p.set_defaults(func=_cmd_config)

    stubs = {
        "scenes": ("generate USD scenes only", "M2"),
        "render": ("render USD scenes with Blender", "M3"),
        "generate": ("generate, render and label a dataset", "M5"),
        "reproduce": ("regenerate one image of a dataset", "M5"),
        "merge": ("merge shard outputs into one dataset", "M6"),
        "k8s-manifest": ("render Kubernetes Job manifests", "M7"),
    }
    for name, (help_text, milestone) in stubs.items():
        p = sub.add_parser(name, help=f"{help_text} (not implemented yet)")
        p.set_defaults(func=_not_yet(milestone))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as err:
        print(err, file=sys.stderr)
        return 1
    except FileNotFoundError as err:
        print(f"usd-synth: file not found: {err.filename}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
