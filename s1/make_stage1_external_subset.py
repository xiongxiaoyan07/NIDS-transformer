#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a small Stage1 external-test subset from paired packet/flow CSV files.

The Stage1 pipeline requires both:
    --external_flow_csv
    --external_packet_csv

This script samples flow rows first, guarantees malicious examples when
available, then streams the large packet CSV in chunks and keeps only packets
whose flow_id is in the sampled flow set.

Example:
    python s1/make_stage1_external_subset.py ^
      --flow_csv dataset/Wednesday-workingHours-stage1_flows.csv ^
      --packet_csv dataset/Wednesday-workingHours-stage1_packets.csv ^
      --out_dir dataset/subsets ^
      --prefix Wednesday-workingHours-external5000 ^
      --num_flows 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_FLOW_CSV = "dataset/Wednesday-workingHours-stage1_flows.csv"
DEFAULT_PACKET_CSV = "dataset/Wednesday-workingHours-stage1_packets.csv"
DEFAULT_OUT_DIR = "dataset/subsets"
DEFAULT_PREFIX = "Wednesday-workingHours-external5000"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(to_jsonable(x) for x in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def binary_labels(label_series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(label_series, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        return (numeric.fillna(0).to_numpy(dtype=np.float64) != 0).astype(np.int64)

    normalized = label_series.astype(str).str.strip().str.upper()
    benign_values = {"0", "BENIGN", "NORMAL", "BACKGROUND"}
    return (~normalized.isin(benign_values)).astype(np.int64).to_numpy()


def read_and_clean_flows(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    flow_csv = Path(args.flow_csv)
    if not flow_csv.exists():
        raise FileNotFoundError(flow_csv)

    print(f"[INFO] reading flows: {flow_csv}")
    flows = normalize_columns(pd.read_csv(flow_csv, low_memory=False))
    require_columns(flows, [args.flow_id_col, args.label_col], "flow csv")

    raw_rows = len(flows)
    flows[args.flow_id_col] = pd.to_numeric(flows[args.flow_id_col], errors="coerce")
    flows[args.label_col] = pd.to_numeric(flows[args.label_col], errors="coerce")
    flows = flows.dropna(subset=[args.flow_id_col, args.label_col]).copy()
    flows[args.flow_id_col] = flows[args.flow_id_col].astype("int64")
    flows[args.label_col] = flows[args.label_col].astype(int)

    duplicate_rows = int(flows[args.flow_id_col].duplicated().sum())
    if args.drop_duplicate_flow_ids:
        flows = flows.drop_duplicates(subset=[args.flow_id_col], keep="first").copy()

    flows = flows.reset_index(drop=False).rename(columns={"index": "__original_row_index"})
    y = binary_labels(flows[args.label_col])

    report = {
        "flow_csv": str(flow_csv),
        "raw_flow_rows": raw_rows,
        "flow_rows_after_required_filter": int(len(flows)),
        "duplicate_flow_id_rows_in_source": duplicate_rows,
        "drop_duplicate_flow_ids": bool(args.drop_duplicate_flow_ids),
        "source_binary_counts_after_clean": {
            int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()
        },
    }
    return flows, report


def compute_target_counts(
    y: np.ndarray,
    num_flows: int,
    min_malicious: int,
    malicious_ratio: Optional[float],
) -> Tuple[int, int]:
    total = len(y)
    num_flows = min(int(num_flows), total)
    n0_available = int((y == 0).sum())
    n1_available = int((y == 1).sum())

    if n1_available == 0 and min_malicious > 0:
        raise ValueError("No malicious/class1 flows are available in the source flow CSV.")

    if malicious_ratio is None:
        n1 = int(round(num_flows * n1_available / max(total, 1)))
        n1 = max(n1, min_malicious if n1_available > 0 else 0)
    else:
        if not 0.0 <= malicious_ratio <= 1.0:
            raise ValueError("--malicious_ratio must be in [0, 1].")
        n1 = int(round(num_flows * malicious_ratio))
        n1 = max(n1, min_malicious if n1_available > 0 else 0)

    n1 = min(n1, n1_available, num_flows)
    n0 = min(num_flows - n1, n0_available)

    # If one class is exhausted, fill remaining slots from the other class.
    remaining = num_flows - n0 - n1
    if remaining > 0:
        add1 = min(remaining, n1_available - n1)
        n1 += add1
        remaining -= add1
    if remaining > 0:
        add0 = min(remaining, n0_available - n0)
        n0 += add0
        remaining -= add0

    if n1 < min_malicious and n1_available >= min_malicious:
        raise ValueError(
            f"Could not satisfy min_malicious={min_malicious}; got {n1}."
        )

    return n0, n1


def sample_flows(flows: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    y = binary_labels(flows[args.label_col])
    n0, n1 = compute_target_counts(
        y=y,
        num_flows=args.num_flows,
        min_malicious=args.min_malicious,
        malicious_ratio=args.malicious_ratio,
    )

    rng = np.random.default_rng(args.seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    chosen_parts: List[np.ndarray] = []
    if n0 > 0:
        chosen_parts.append(rng.choice(idx0, size=n0, replace=False))
    if n1 > 0:
        chosen_parts.append(rng.choice(idx1, size=n1, replace=False))

    chosen = np.concatenate(chosen_parts)
    rng.shuffle(chosen)
    sampled = flows.iloc[chosen].copy()

    if args.output_order == "original":
        sampled = sampled.sort_values("__original_row_index", kind="mergesort")
    elif args.output_order == "time":
        if args.time_col not in sampled.columns:
            raise ValueError(f"--output_order time requires column {args.time_col!r}.")
        sampled = sampled.sort_values(args.time_col, kind="mergesort")
    elif args.output_order != "random":
        raise ValueError("--output_order must be one of: random, original, time")

    sampled = sampled.drop(columns=["__original_row_index"]).reset_index(drop=True)
    y_sample = binary_labels(sampled[args.label_col])
    report = {
        "requested_num_flows": int(args.num_flows),
        "requested_min_malicious": int(args.min_malicious),
        "requested_malicious_ratio": args.malicious_ratio,
        "sampled_flow_rows_before_packet_check": int(len(sampled)),
        "sampled_binary_counts_before_packet_check": {
            int(k): int(v) for k, v in pd.Series(y_sample).value_counts().sort_index().items()
        },
        "output_order": args.output_order,
    }
    return sampled, report


def stream_filter_packets(
    packet_csv: str | Path,
    out_packet_csv: str | Path,
    selected_flow_ids: set[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    packet_csv = Path(packet_csv)
    out_packet_csv = Path(out_packet_csv)
    if not packet_csv.exists():
        raise FileNotFoundError(packet_csv)

    print(f"[INFO] streaming packets: {packet_csv}")
    print(f"[INFO] selected flow_ids: {len(selected_flow_ids):,}")

    total_packet_rows = 0
    kept_packet_rows = 0
    chunks = 0
    found_flow_ids: set[int] = set()
    wrote_header = False

    reader = pd.read_csv(packet_csv, chunksize=args.packet_chunksize, low_memory=False)
    for chunk in reader:
        chunks += 1
        chunk = normalize_columns(chunk)
        require_columns(chunk, [args.flow_id_col], "packet csv")
        total_packet_rows += len(chunk)

        flow_ids = pd.to_numeric(chunk[args.flow_id_col], errors="coerce").fillna(-1).astype("int64")
        mask = flow_ids.isin(selected_flow_ids)
        if not mask.any():
            if chunks % args.progress_every_chunks == 0:
                print(
                    f"[INFO] chunks={chunks}, scanned={total_packet_rows:,}, kept={kept_packet_rows:,}"
                )
            continue

        kept = chunk.loc[mask].copy()
        kept_flow_ids = pd.to_numeric(kept[args.flow_id_col], errors="coerce").dropna().astype("int64")
        found_flow_ids.update(int(x) for x in kept_flow_ids.unique().tolist())
        kept_packet_rows += len(kept)

        kept.to_csv(
            out_packet_csv,
            index=False,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
        )
        wrote_header = True

        if chunks % args.progress_every_chunks == 0:
            print(
                f"[INFO] chunks={chunks}, scanned={total_packet_rows:,}, kept={kept_packet_rows:,}"
            )

    if not wrote_header:
        raise ValueError("No packets were found for the sampled flow_ids.")

    missing_flow_ids = selected_flow_ids - found_flow_ids
    return {
        "packet_csv": str(packet_csv),
        "out_packet_csv": str(out_packet_csv),
        "packet_chunks_read": int(chunks),
        "packet_rows_scanned": int(total_packet_rows),
        "packet_rows_kept": int(kept_packet_rows),
        "sampled_flow_ids_with_packets": int(len(found_flow_ids)),
        "sampled_flow_ids_missing_packets": int(len(missing_flow_ids)),
        "missing_flow_ids_preview": sorted(list(missing_flow_ids))[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample paired Stage1 packet/flow CSVs for external testing."
    )
    parser.add_argument("--flow_csv", default=DEFAULT_FLOW_CSV)
    parser.add_argument("--packet_csv", default=DEFAULT_PACKET_CSV)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--num_flows", type=int, default=5000)
    parser.add_argument("--min_malicious", type=int, default=1)
    parser.add_argument(
        "--malicious_ratio",
        type=float,
        default=None,
        help="Optional target class1 ratio. If omitted, preserve source binary ratio.",
    )
    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--flow_id_col", default="flow_id")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--time_col", default="flow_start_timestamp_us")
    parser.add_argument(
        "--output_order",
        choices=["random", "original", "time"],
        default="original",
    )
    parser.add_argument(
        "--drop_duplicate_flow_ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep this enabled for Stage1 external tests.",
    )
    parser.add_argument("--packet_chunksize", type=int, default=500_000)
    parser.add_argument("--progress_every_chunks", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    out_dir = Path(args.out_dir)
    out_flow_csv = out_dir / f"{args.prefix}-stage1_flows.csv"
    out_packet_csv = out_dir / f"{args.prefix}-stage1_packets.csv"
    out_report_json = out_dir / f"{args.prefix}-subset_report.json"

    flows, source_report = read_and_clean_flows(args)
    sampled_flows, sample_report = sample_flows(flows, args)
    selected_flow_ids = set(int(x) for x in sampled_flows[args.flow_id_col].tolist())

    packet_report = stream_filter_packets(
        packet_csv=args.packet_csv,
        out_packet_csv=out_packet_csv,
        selected_flow_ids=selected_flow_ids,
        args=args,
    )

    found_flow_ids_count = packet_report["sampled_flow_ids_with_packets"]
    if found_flow_ids_count != len(selected_flow_ids):
        # Drop flows that have no packets so Stage1 read_stage1_csvs keeps the same set.
        packet_ids_in_output = set()
        for chunk in pd.read_csv(out_packet_csv, usecols=[args.flow_id_col], chunksize=args.packet_chunksize):
            ids = pd.to_numeric(chunk[args.flow_id_col], errors="coerce").dropna().astype("int64")
            packet_ids_in_output.update(int(x) for x in ids.unique().tolist())
        sampled_flows = sampled_flows[sampled_flows[args.flow_id_col].isin(packet_ids_in_output)].copy()

    sampled_flows.to_csv(out_flow_csv, index=False)

    y_final = binary_labels(sampled_flows[args.label_col])
    final_report = {
        **source_report,
        **sample_report,
        **packet_report,
        "out_flow_csv": str(out_flow_csv),
        "out_packet_csv": str(out_packet_csv),
        "final_flow_rows": int(len(sampled_flows)),
        "final_binary_counts": {
            int(k): int(v) for k, v in pd.Series(y_final).value_counts().sort_index().items()
        },
    }
    save_json(final_report, out_report_json)

    print("\n[DONE] subset created")
    print(f"  flow_csv   : {out_flow_csv.resolve()}")
    print(f"  packet_csv : {out_packet_csv.resolve()}")
    print(f"  report     : {out_report_json.resolve()}")
    print(f"  final flows: {len(sampled_flows):,}")
    print(f"  final label counts: {final_report['final_binary_counts']}")


if __name__ == "__main__":
    main()
