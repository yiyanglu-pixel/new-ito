import csv
import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np


def main(args):
    rows = []
    for path in args.metrics:
        row = flatten(json.load(open(path)))
        row["metrics_path"] = os.path.realpath(path)
        rows.append(row)

    if not rows:
        raise ValueError("no metrics files provided")

    os.makedirs(args.out_dir, exist_ok=True)
    raw_csv = os.path.join(args.out_dir, "ala2_metrics_raw.csv")
    summary_csv = os.path.join(args.out_dir, "ala2_metrics_summary.csv")
    summary_json = os.path.join(args.out_dir, "ala2_metrics_summary.json")

    write_csv(raw_csv, rows)
    summary_rows = summarise(rows, args.group_by.split(","))
    write_csv(summary_csv, summary_rows)
    json.dump(summary_rows, open(summary_json, "w"), indent=4)

    print(f"raw metrics: {raw_csv}")
    print(f"summary: {summary_csv}")


def flatten(obj, prefix=""):
    out = {}
    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten(value, name))
        else:
            out[name] = value
    return out


def summarise(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups[key].append(row)

    summary_rows = []
    for key, group in sorted(groups.items()):
        out = {k: v for k, v in zip(group_keys, key)}
        out["n"] = len(group)
        numeric_keys = sorted(
            {
                k
                for row in group
                for k, v in row.items()
                if is_number(v) and k not in group_keys
            }
        )
        for metric in numeric_keys:
            values = np.array(
                [float(row[metric]) for row in group if is_number(row.get(metric))]
            )
            if len(values) == 0:
                continue
            out[f"{metric}.mean"] = float(values.mean())
            out[f"{metric}.std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "metrics",
        nargs="+",
        help="One or more metrics.json files produced by analyse_trajs.py/analyse_bridge.py.",
    )
    parser.add_argument(
        "--out_dir",
        default="storage/analysis_summary",
        help="Directory where raw and grouped summary tables are written.",
    )
    parser.add_argument(
        "--group_by",
        default="model,grid,ref_lag",
        help="Comma-separated fields used for mean/std grouping.",
    )
    main(parser.parse_args())
