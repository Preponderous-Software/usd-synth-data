"""``usd-synth`` command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import ConfigError, load_config
from .render import RenderError
from .seeds import check_seed


def _cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.resolved:
        print(json.dumps(cfg.resolved(), indent=2, sort_keys=True))
    print(f"config OK: {args.config} sha256={cfg.config_hash()}")
    return 0


def _cmd_scenes(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .scenegen import write_scene

    cfg = load_config(args.config)
    check_seed(args.seed)
    config_dir = Path(args.config).resolve().parent
    for index in range(args.start, args.start + args.count):
        path = write_scene(cfg, args.seed, index, args.out, config_dir)
        print(path)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .render import find_blender, render_batch, scene_spec

    cfg = load_config(args.config)
    src = Path(args.scenes)
    files = sorted(src.glob("*.usda")) if src.is_dir() else [src]
    if not files:
        print(f"usd-synth render: no .usda files in {src}", file=sys.stderr)
        return 1
    blender = find_blender(args.blender)
    specs = [scene_spec(f, Path(args.out) / f.stem) for f in files]
    size = cfg.render.batch_size
    for start in range(0, len(specs), size):
        result = render_batch(cfg, specs[start : start + size], blender, threads=args.threads)
        for info in result.scenes:
            t = info["timings"]
            print(f"{info['usd']}: render {t['render']:.2f}s ({info['device']})")
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

    p = sub.add_parser("scenes", help="generate USD scenes only")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--seed", type=int, required=True, help="dataset seed (>= 0)")
    p.add_argument("--count", type=int, required=True, help="number of scenes")
    p.add_argument("--start", type=int, default=0, help="first scene index (default 0)")
    p.add_argument("--out", required=True, help="directory for scene_NNNNNN.usda files")
    p.set_defaults(func=_cmd_scenes)

    p = sub.add_parser("render", help="render USD scenes with Blender (RGB + depth + index)")
    p.add_argument("--config", required=True, help="config providing render settings")
    p.add_argument("--scenes", required=True, help="a .usda file or a directory of them")
    p.add_argument("--out", required=True, help="output directory (one subdirectory per scene)")
    p.add_argument("--blender", help="Blender binary (default: $USD_SYNTH_BLENDER or PATH)")
    p.add_argument("--threads", type=int, help="override render.threads")
    p.set_defaults(func=_cmd_render)

    stubs = {
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
    except (RenderError, ValueError) as err:  # e.g. Blender failed, an out-of-range seed
        print(f"usd-synth: {err}", file=sys.stderr)
        return 1
    except FileNotFoundError as err:
        print(f"usd-synth: file not found: {err.filename}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
