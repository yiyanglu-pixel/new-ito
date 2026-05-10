"""Aggregate metrics.json files produced by analyse_trajs / analyse_bridge.

Usage:
    python scripts/summarise_ala2_metrics.py 'storage_*/*/*/metrics.json' \\
        --out_dir storage_eval_summary

Groups by (grid, model_type) and reports mean / std across seeds.
"""
import csv
import glob as globlib
import json
import os
import statistics
from argparse import ArgumentParser
from collections import defaultdict


KEY_FIELDS = ("grid", "model_type")
SCALAR_FIELDS = ("vamp2", "ref_vamp2", "ref_lag")
NESTED_SCALAR_FIELDS = (
    ("midpoint", "phi_circular_rmse"),
    ("midpoint", "psi_circular_rmse"),
)


def load_metrics(patterns):
    """Resolve user-supplied glob patterns to a flat list of metrics dicts."""
    paths = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            paths.extend(sorted(globlib.glob(pattern)))
        elif os.path.isdir(pattern):
            paths.extend(sorted(globlib.glob(os.path.join(pattern, "**", "metrics.json"), recursive=True)))
        elif os.path.isfile(pattern):
            paths.append(pattern)
    paths = [p for p in paths if os.path.basename(p) == "metrics.json"]

    metrics = []
    for p in paths:
        with open(p) as f:
            m = json.load(f)
        m["_path"] = p
        metrics.append(m)
    return metrics


def get_nested(d, path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def summarise(metrics):
    groups = defaultdict(list)
    for m in metrics:
        key = tuple(m.get(k, "") for k in KEY_FIELDS)
        groups[key].append(m)

    rows = []
    for key, runs in sorted(groups.items()):
        row = dict(zip(KEY_FIELDS, key))
        row["n_seeds"] = len(runs)
        row["seeds"] = sorted(
            r.get("seed") for r in runs if r.get("seed") is not None
        )
        for field in SCALAR_FIELDS:
            vals = [r[field] for r in runs if r.get(field) is not None]
            row[f"{field}_mean"] = statistics.mean(vals) if vals else None
            row[f"{field}_std"] = (
                statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None
            )
        for path in NESTED_SCALAR_FIELDS:
            vals = [v for r in runs if (v := get_nested(r, path)) is not None]
            label = ".".join(path)
            row[f"{label}_mean"] = statistics.mean(vals) if vals else None
            row[f"{label}_std"] = (
                statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None
            )
        rows.append(row)
    return rows


def write_csv(rows, out_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = {k: ("" if v is None else v) for k, v in row.items()}
            writer.writerow(row_out)


def write_markdown(rows, out_path):
    if not rows:
        return

    headers = [
        "grid",
        "model_type",
        "n_seeds",
        "vamp2_mean",
        "vamp2_std",
        "ref_vamp2_mean",
        "ref_lag_mean",
        "midpoint.phi_circular_rmse_mean",
        "midpoint.psi_circular_rmse_mean",
    ]

    def fmt(v):
        if v is None or v == "":
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(args):
    metrics = load_metrics(args.patterns)
    if not metrics:
        raise SystemExit(f"no metrics.json found under {args.patterns}")

    print(f"loaded {len(metrics)} metrics.json files")

    rows = summarise(metrics)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "summary.csv")
    md_path = os.path.join(args.out_dir, "summary.md")
    raw_path = os.path.join(args.out_dir, "raw.json")

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    json.dump(metrics, open(raw_path, "w"), indent=4, default=str)

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote {raw_path}")

    with open(md_path) as f:
        print()
        print(f.read())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "patterns",
        nargs="+",
        help="One or more metrics.json paths, directories, or glob patterns. "
        "Quote globs in your shell so the shell doesn't expand them prematurely.",
    )
    parser.add_argument(
        "--out_dir",
        default="storage_eval_summary",
        help="Directory for summary.csv, summary.md, and raw.json.",
    )

    main(parser.parse_args())
