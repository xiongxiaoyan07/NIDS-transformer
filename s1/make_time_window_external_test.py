#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a small chronological external-test subset for Stage1.

Why this exists:
    stage1.pipeline.build_dataloaders uses external_packet_csv + external_flow_csv
    together when has_external_test is enabled. Therefore a flow-only subset is not
    enough: the packet CSV must be filtered to the same selected flow_id values.

Default behavior:
    1. Read and Stage1-clean the flow CSV.
    2. Sort flows by flow_start_timestamp_us.
    3. Select one contiguous time-ordered window of about 5000 unique flows.
    4. Ensure the window contains malicious label=1 and benign label=0.
    5. Write selected flows.
    6. Filter the huge packet CSV in chunks and write matching packets only.

Example:
    python s1/make_time_window_external_test.py ^
      --flow_csv dataset/Wednesday-workingHours-stage1_flows.csv ^
      --packet_csv dataset/Wednesday-workingHours-stage1_packets.csv ^
      --out_dir dataset/wednesday_external_5000 ^
      --n_flows 5000 ^
      --selection balanced

Use the outputs in Stage1:
    --external_flow_csv dataset/wednesday_external_5000/Wednesday-workingHours-external5000-stage1_flows.csv
    --external_packet_csv dataset/wednesday_external_5000/Wednesday-workingHours-external5000-stage1_packets.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_FLOW_CSV = "dataset/Wednesday-workingHours-stage1_flows.csv"
DEFAULT_PACKET_CSV = "dataset/Wednesday-workingHours-stage1_packets.csv"
DEFAULT_OUT_DIR = "dataset/wednesday_external_5000"
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


def write_lines(lines: Iterable[Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def label_to_binary(label_series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(label_series, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        return (numeric.fillna(0).to_numpy(dtype=np.float64) != 0).astype(np.int64)

    text = label_series.astype(str).str.strip().str.upper()
    benign_values = {"0", "BENIGN", "NORMAL", "BACKGROUND"}
    return (~text.isin(benign_values)).astype(np.int64).to_numpy()


def clean_flows_like_stage1(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = normalize_columns(df)
    if "record_type" in df.columns:
        df = df.drop(columns=["record_type"])

    ensure_columns(
        df,
        [args.flow_id_col, args.time_col, args.label_col],
        "flow csv",
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df[args.flow_id_col] = pd.to_numeric(df[args.flow_id_col], errors="coerce")
    df[args.time_col] = pd.to_numeric(df[args.time_col], errors="coerce")
    df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(int)

    before = len(df)
    df = df[
        (df[args.flow_id_col] != 0)
        & (df[args.time_col] != 0)
        & df[args.flow_id_col].notna()
        & df[args.time_col].notna()
    ].copy()
    dropped_invalid = before - len(df)

    df[args.flow_id_col] = df[args.flow_id_col].astype("int64")
    df[args.time_col] = df[args.time_col].astype("int64")
    df = df.sort_values(args.time_col, kind="mergesort").reset_index(drop=True)

    duplicate_count = int(df[args.flow_id_col].duplicated().sum())
    if args.deduplicate_flow_ids:
        df = df.drop_duplicates(args.flow_id_col, keep="first").reset_index(drop=True)

    df.attrs["dropped_invalid_stage1_clean"] = int(dropped_invalid)
    df.attrs["duplicate_flow_id_rows_before_dedup"] = duplicate_count
    return df


def load_and_prepare_flows(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = Path(args.flow_csv)
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[INFO] loading flow CSV: {path}")
    raw = pd.read_csv(path, low_memory=False)
    raw_rows, raw_cols = raw.shape
    flows = clean_flows_like_stage1(raw, args)

    y = label_to_binary(flows[args.label_col])
    counts = {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()}

    report = {
        "flow_csv": str(path),
        "raw_rows": int(raw_rows),
        "raw_columns": int(raw_cols),
        "rows_after_stage1_clean": int(len(flows)),
        "dropped_invalid_stage1_clean": int(flows.attrs.get("dropped_invalid_stage1_clean", 0)),
        "duplicate_flow_id_rows_before_dedup": int(flows.attrs.get("duplicate_flow_id_rows_before_dedup", 0)),
        "deduplicate_flow_ids": bool(args.deduplicate_flow_ids),
        "rows_after_dedup": int(len(flows)),
        "label_counts_after_clean": counts,
        "attack_ratio_after_clean": float(y.mean()) if len(y) else 0.0,
    }

    print(
        "[DATA] flows after clean={:,}, labels={}, attack_ratio={:.6f}".format(
            len(flows),
            counts,
            report["attack_ratio_after_clean"],
        )
    )
    print(
        "[DATA] duplicate flow_id rows before dedup={:,}".format(
            report["duplicate_flow_id_rows_before_dedup"]
        )
    )
    return flows, report


def rolling_window_counts(y: np.ndarray, window_size: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    cumsum = np.concatenate([[0], np.cumsum(y)])
    return cumsum[window_size:] - cumsum[:-window_size]


def choose_time_window(flows: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if len(flows) == 0:
        raise ValueError("No flows left after cleaning.")

    n = min(int(args.n_flows), len(flows))
    if n <= 0:
        raise ValueError("--n_flows must be positive.")

    y = label_to_binary(flows[args.label_col])
    if y.sum() == 0:
        raise ValueError("The flow CSV contains no malicious labels after cleaning.")

    pos_counts = rolling_window_counts(y, n)
    benign_counts = n - pos_counts
    valid = (pos_counts >= args.min_malicious) & (benign_counts >= args.min_benign)

    if not valid.any():
        best_idx = int(np.argmax(pos_counts))
        raise ValueError(
            "No contiguous time window satisfies "
            f"min_malicious={args.min_malicious}, min_benign={args.min_benign}. "
            f"Best malicious count was {int(pos_counts[best_idx])}/{n} "
            f"at start index {best_idx}."
        )

    valid_indices = np.flatnonzero(valid)
    selection = str(args.selection).lower()

    if selection == "first_with_malicious":
        start = int(valid_indices[0])
    elif selection == "max_malicious":
        scores = pos_counts[valid_indices]
        start = int(valid_indices[int(np.argmax(scores))])
    elif selection == "balanced":
        target = args.target_attack_ratio
        if target is None:
            target = float(y.mean())
        ratios = pos_counts[valid_indices] / float(n)
        distance = np.abs(ratios - float(target))
        best_distance = float(distance.min())
        candidates = valid_indices[np.flatnonzero(distance == best_distance)]
        start = int(candidates[0])
    elif selection == "around_first_malicious":
        first_pos = int(np.flatnonzero(y == 1)[0])
        start = max(0, min(first_pos - n // 2, len(flows) - n))
        if not valid[start]:
            valid_before = valid_indices[valid_indices <= start]
            valid_after = valid_indices[valid_indices >= start]
            candidates = []
            if len(valid_before):
                candidates.append(int(valid_before[-1]))
            if len(valid_after):
                candidates.append(int(valid_after[0]))
            start = min(candidates, key=lambda idx: abs(idx - start))
    else:
        raise ValueError(f"Unsupported selection: {args.selection!r}")

    end = start + n
    selected = flows.iloc[start:end].copy()
    selected_y = label_to_binary(selected[args.label_col])
    selected_counts = {
        int(k): int(v)
        for k, v in pd.Series(selected_y).value_counts().sort_index().items()
    }

    info = {
        "selection": args.selection,
        "requested_n_flows": int(args.n_flows),
        "selected_n_flows": int(len(selected)),
        "start_index_after_time_sort": int(start),
        "end_index_exclusive_after_time_sort": int(end),
        "selected_label_counts": selected_counts,
        "selected_attack_ratio": float(selected_y.mean()) if len(selected_y) else 0.0,
        "selected_time_start_us": int(selected[args.time_col].iloc[0]),
        "selected_time_end_us": int(selected[args.time_col].iloc[-1]),
        "selected_time_duration_us": int(selected[args.time_col].iloc[-1] - selected[args.time_col].iloc[0]),
        "min_malicious": int(args.min_malicious),
        "min_benign": int(args.min_benign),
    }

    print(
        "[SELECT] rows={:,}, label_counts={}, attack_ratio={:.6f}, time=[{}, {}]".format(
            len(selected),
            selected_counts,
            info["selected_attack_ratio"],
            info["selected_time_start_us"],
            info["selected_time_end_us"],
        )
    )
    return selected, info


def clean_packet_chunk_like_stage1(chunk: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    chunk = normalize_columns(chunk)
    ensure_columns(chunk, [args.flow_id_col, args.packet_time_col], "packet csv")
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk[args.flow_id_col] = pd.to_numeric(chunk[args.flow_id_col], errors="coerce")
    chunk[args.packet_time_col] = pd.to_numeric(chunk[args.packet_time_col], errors="coerce")
    return chunk[
        (chunk[args.flow_id_col] != 0)
        & (chunk[args.packet_time_col] != 0)
        & chunk[args.flow_id_col].notna()
        & chunk[args.packet_time_col].notna()
    ].copy()


def filter_packets_by_flow_ids(
    packet_csv: str | Path,
    out_packet_csv: str | Path,
    selected_flow_ids: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    packet_path = Path(packet_csv)
    if not packet_path.exists():
        raise FileNotFoundError(packet_path)

    out_path = Path(out_packet_csv)
    selected_set = {int(x) for x in selected_flow_ids}
    matched_flow_ids = set()
    packet_rows = 0
    raw_rows_seen = 0
    chunks_seen = 0
    first_write = True

    print(f"[INFO] filtering packet CSV in chunks: {packet_path}")
    print(f"[INFO] writing selected packets to: {out_path}")

    for chunk in pd.read_csv(packet_path, low_memory=False, chunksize=args.chunksize):
        chunks_seen += 1
        raw_rows_seen += len(chunk)
        chunk = clean_packet_chunk_like_stage1(chunk, args)
        flow_ids = pd.to_numeric(chunk[args.flow_id_col], errors="coerce")
        mask = flow_ids.astype("Int64").isin(selected_set)
        sub = chunk.loc[mask].copy()

        if len(sub) > 0:
            sub[args.flow_id_col] = sub[args.flow_id_col].astype("int64")
            matched_flow_ids.update(sub[args.flow_id_col].unique().astype("int64").tolist())
            packet_rows += len(sub)
            sub.to_csv(
                out_path,
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )
            first_write = False

        if chunks_seen % args.progress_every_chunks == 0:
            print(
                "[INFO] chunks={}, raw_rows_seen={:,}, selected_packets={:,}, matched_flows={:,}".format(
                    chunks_seen,
                    raw_rows_seen,
                    packet_rows,
                    len(matched_flow_ids),
                )
            )

    if first_write:
        # Write an empty CSV with original header if no packet matched.
        header = pd.read_csv(packet_path, nrows=0)
        header = normalize_columns(header)
        header.to_csv(out_path, index=False)

    missing_flow_ids = sorted(selected_set - matched_flow_ids)
    packet_info = {
        "packet_csv": str(packet_path),
        "out_packet_csv": str(out_path),
        "chunks_seen": int(chunks_seen),
        "raw_packet_rows_seen": int(raw_rows_seen),
        "selected_packet_rows": int(packet_rows),
        "matched_flow_count": int(len(matched_flow_ids)),
        "selected_flow_count": int(len(selected_set)),
        "missing_packet_flow_count": int(len(missing_flow_ids)),
        "missing_packet_flow_ids_head": missing_flow_ids[:20],
    }
    print(
        "[PACKETS] selected_packets={:,}, matched_flows={:,}/{:,}, missing_flows={:,}".format(
            packet_rows,
            len(matched_flow_ids),
            len(selected_set),
            len(missing_flow_ids),
        )
    )
    return packet_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a small chronological external-test CSV pair for Stage1."
    )
    parser.add_argument("--flow_csv", default=DEFAULT_FLOW_CSV)
    parser.add_argument("--packet_csv", default=DEFAULT_PACKET_CSV)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)

    parser.add_argument("--flow_id_col", default="flow_id")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--time_col", default="flow_start_timestamp_us")
    parser.add_argument("--packet_time_col", default="timestamp_us")

    parser.add_argument("--n_flows", type=int, default=5000)
    parser.add_argument(
        "--selection",
        choices=[
            "balanced",
            "first_with_malicious",
            "around_first_malicious",
            "max_malicious",
        ],
        default="balanced",
        help="How to choose the contiguous time-sorted flow window.",
    )
    parser.add_argument("--target_attack_ratio", type=float, default=None)
    parser.add_argument("--min_malicious", type=int, default=1)
    parser.add_argument("--min_benign", type=int, default=1)
    parser.add_argument("--deduplicate_flow_ids", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--progress_every_chunks", type=int, default=5)
    parser.add_argument(
        "--skip_packet_filter",
        action="store_true",
        help="Write only the flow subset. Useful for a quick dry run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    out_flow_csv = out_dir / f"{args.prefix}-stage1_flows.csv"
    out_packet_csv = out_dir / f"{args.prefix}-stage1_packets.csv"
    out_flow_ids = out_dir / f"{args.prefix}-flow_ids.txt"
    out_report = out_dir / f"{args.prefix}-report.json"

    flows, flow_report = load_and_prepare_flows(args)
    selected_flows, selection_report = choose_time_window(flows, args)

    selected_flow_ids = selected_flows[args.flow_id_col].astype("int64").tolist()
    selected_flows.to_csv(out_flow_csv, index=False)
    write_lines(selected_flow_ids, out_flow_ids)

    packet_report: Optional[Dict[str, Any]] = None
    if not args.skip_packet_filter:
        packet_report = filter_packets_by_flow_ids(
            packet_csv=args.packet_csv,
            out_packet_csv=out_packet_csv,
            selected_flow_ids=selected_flow_ids,
            args=args,
        )

    report = {
        "args": vars(args),
        "flow_report": flow_report,
        "selection_report": selection_report,
        "outputs": {
            "external_flow_csv": str(out_flow_csv),
            "external_packet_csv": str(out_packet_csv) if not args.skip_packet_filter else None,
            "flow_ids": str(out_flow_ids),
            "report": str(out_report),
        },
        "packet_report": packet_report,
        "stage1_usage": {
            "external_flow_csv_arg": str(out_flow_csv),
            "external_packet_csv_arg": str(out_packet_csv),
        },
    }
    save_json(report, out_report)

    print("\n[DONE] external subset created")
    print(f"  flow_csv   : {out_flow_csv}")
    if args.skip_packet_filter:
        print("  packet_csv : skipped")
    else:
        print(f"  packet_csv : {out_packet_csv}")
    print(f"  report     : {out_report}")
    print("\nUse with Stage1:")
    print(f"  --external_flow_csv {out_flow_csv}")
    print(f"  --external_packet_csv {out_packet_csv}")


if __name__ == "__main__":
    main()
