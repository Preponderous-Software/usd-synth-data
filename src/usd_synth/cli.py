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


def _shard_worker(job: dict) -> tuple[int, int]:
    """Run one shard in a worker process (``--workers``); returns (rendered, skipped)."""
    from pathlib import Path

    from .pipeline import run_shard
    from .storage import open_sink

    cfg = load_config(job["config"])
    result = run_shard(
        cfg, Path(job["config_dir"]), job["seed"], job["indices"], open_sink(job["out"]),
        job["blender"], shard=job["shard"], threads=job["threads"], batch_size=job["batch_size"],
    )  # fmt: skip
    return len(result.rendered), len(result.skipped)


def _validate_and_report(sink, merged: dict) -> int:
    import tempfile
    from pathlib import Path

    from .coco import validate_coco
    from .throughput import summary_line

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "annotations.json"
        sink.get_file(sink.dataset_prefix + "annotations.json", local)
        errors = validate_coco(local)
    for e in errors:
        print(f"invalid COCO: {e}", file=sys.stderr)
    coco = merged["coco"]
    print(
        f"dataset: {len(coco['images'])} images, {len(coco['annotations'])} annotations -> "
        f"{sink.describe(sink.dataset_prefix)}"
    )
    print(summary_line(merged["throughput"]))
    return 1 if errors else 0


def _cmd_generate(args: argparse.Namespace) -> int:
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    from .merge import merge
    from .pipeline import prepare_shard, run_shard
    from .render import find_blender
    from .storage import open_sink, shard_prefix

    cfg = load_config(args.config)
    check_seed(args.seed)
    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if (args.shard_index is None) != (args.shard_count is None):
        raise ValueError("--shard-index and --shard-count must be given together")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    sink = open_sink(args.out)
    config_dir = Path(args.config).resolve().parent
    blender = find_blender(args.blender)
    threads = args.threads if args.threads is not None else cfg.render.threads

    if args.shard_index is not None:  # one shard only (e.g. one Kubernetes pod); merge later
        k, n = args.shard_index, args.shard_count
        rng = prepare_shard(sink, cfg, config_dir, args.seed, args.count, n, k)
        result = run_shard(
            cfg, config_dir, args.seed, rng, sink, blender,
            shard=k, threads=threads, batch_size=args.batch_size,
        )  # fmt: skip
        print(
            f"shard {k}/{n}: scenes [{rng.start}, {rng.stop}): {len(result.rendered)} rendered, "
            f"{len(result.skipped)} already done -> {sink.describe(shard_prefix(k))}"
        )
        return 0

    n = args.workers
    if n > 1 and not threads:  # 0 = Blender decides, which would oversubscribe the CPU
        threads = max(1, (os.cpu_count() or n) // n)
    ranges = [prepare_shard(sink, cfg, config_dir, args.seed, args.count, n, k) for k in range(n)]
    if n == 1:
        run_shard(
            cfg, config_dir, args.seed, ranges[0], sink, blender,
            shard=0, threads=threads, batch_size=args.batch_size,
        )  # fmt: skip
    else:
        jobs = [
            {"config": str(args.config), "config_dir": str(config_dir), "seed": args.seed,
             "indices": list(r), "out": str(args.out), "blender": blender, "shard": k,
             "threads": threads, "batch_size": args.batch_size}
            for k, r in enumerate(ranges)
        ]  # fmt: skip
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
            list(pool.map(_shard_worker, jobs))
    return _validate_and_report(sink, merge(sink))


def _cmd_merge(args: argparse.Namespace) -> int:
    from .merge import merge
    from .storage import open_sink

    sink = open_sink(args.out)
    return _validate_and_report(sink, merge(sink))


def _cmd_reproduce(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .render import find_blender
    from .reproduce import reproduce

    report = reproduce(
        Path(args.dataset), args.image_id, Path(args.out), find_blender(args.blender),
        threads=args.threads,
    )  # fmt: skip
    print(
        f"image {report['image_id']} (scene {report['scene_index']}): "
        f"usda identical={report['usda_identical']}, "
        f"annotations identical={report['annotations_identical']}, "
        f"RGB mean abs diff={report['rgb_mean_abs_diff']:.4f} "
        f"(threshold {report['rgb_threshold']}) -> {'OK' if report['ok'] else 'MISMATCH'}"
    )
    return 0 if report["ok"] else 1


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

    p = sub.add_parser("generate", help="generate, render and label a dataset")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--seed", type=int, required=True, help="dataset seed (>= 0)")
    p.add_argument("--count", type=int, required=True, help="number of images")
    p.add_argument("--out", required=True, help="dataset directory (re-run to resume)")
    p.add_argument("--blender", help="Blender binary (default: $USD_SYNTH_BLENDER or PATH)")
    p.add_argument("--threads", type=int, help="override render.threads")
    p.add_argument("--batch-size", type=int, help="override render.batch_size")
    p.add_argument("--workers", type=int, default=1, help="local worker processes (shards)")
    p.add_argument("--shard-index", type=int, help="run only this shard (then use merge)")
    p.add_argument("--shard-count", type=int, help="total number of shards")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("reproduce", help="regenerate one image of a dataset and compare")
    p.add_argument("--dataset", required=True, help="dataset directory with manifest.json")
    p.add_argument("--image-id", type=int, required=True, help="COCO image id")
    p.add_argument("--out", required=True, help="directory for the regenerated image")
    p.add_argument("--blender", help="Blender binary (default: $USD_SYNTH_BLENDER or PATH)")
    p.add_argument("--threads", type=int, help="override render.threads")
    p.set_defaults(func=_cmd_reproduce)

    p = sub.add_parser("merge", help="merge shard outputs into one dataset")
    p.add_argument("--out", required=True, help="run directory holding shards/ (as for generate)")
    p.set_defaults(func=_cmd_merge)

    stubs = {
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
    except (RenderError, ValueError) as err:  # Blender failed, bad seed, MergeError, ...
        print(f"usd-synth: {err}", file=sys.stderr)
        return 1
    except FileNotFoundError as err:
        print(f"usd-synth: file not found: {err.filename}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
