#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create multiple non-overlapping, chronological Stage1 external-test pairs.

Flows are sorted by flow_start_timestamp_us and split into small windows.  The
large packet CSV is scanned only once; each packet is routed by flow_id to the
matching window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create multiple chronological flow/packet CSV pairs."
    )
    parser.add_argument(
        "--flow_csv",
        default="dataset/Wednesday-workingHours-stage1_flows.csv",
    )
    parser.add_argument(
        "--packet_csv",
        default="dataset/Wednesday-workingHours-stage1_packets.csv",
    )
    parser.add_argument("--out_dir", default="dataset/wednesday_external_windows")
    parser.add_argument("--prefix", default="Wednesday-workingHours-external")
    parser.add_argument("--n_windows", type=int, default=9)
    parser.add_argument("--flows_per_window", type=int, default=30000)
    parser.add_argument(
        "--placement",
        choices=["even", "consecutive"],
        default="even",
        help=(
            "even: spread windows across the complete timeline; "
            "consecutive: take adjacent windows from --start_fraction."
        ),
    )
    parser.add_argument(
        "--start_fraction",
        type=float,
        default=0.0,
        help="Timeline starting position in [0,1); mainly used by consecutive.",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--progress_every_chunks", type=int, default=5)
    return parser.parse_args()


def load_flows(path: Path) -> pd.DataFrame:
    flows = pd.read_csv(path, low_memory=False)
    flows.columns = [str(c).strip() for c in flows.columns]
    required = {"flow_id", "flow_start_timestamp_us", "label"}
    missing = sorted(required - set(flows.columns))
    if missing:
        raise ValueError(f"flow CSV missing columns: {missing}")

    flows["flow_id"] = pd.to_numeric(flows["flow_id"], errors="coerce")
    flows["flow_start_timestamp_us"] = pd.to_numeric(
        flows["flow_start_timestamp_us"], errors="coerce"
    )
    flows = flows[
        flows["flow_id"].notna()
        & flows["flow_start_timestamp_us"].notna()
        & (flows["flow_id"] != 0)
        & (flows["flow_start_timestamp_us"] != 0)
    ].copy()
    flows["flow_id"] = flows["flow_id"].astype("int64")
    flows["flow_start_timestamp_us"] = flows[
        "flow_start_timestamp_us"
    ].astype("int64")
    flows = flows.drop_duplicates("flow_id", keep="first")
    return flows.sort_values(
        ["flow_start_timestamp_us", "flow_id"], kind="mergesort"
    ).reset_index(drop=True)


def choose_starts(total: int, args: argparse.Namespace) -> list[int]:
    size = args.flows_per_window
    count = args.n_windows
    if size <= 0 or count <= 0:
        raise ValueError("--flows_per_window and --n_windows must be positive")
    if not 0.0 <= args.start_fraction < 1.0:
        raise ValueError("--start_fraction must be in [0, 1)")
    if count * size > total:
        raise ValueError(
            f"Need {count * size:,} unique flows, but only {total:,} are available."
        )

    max_start = total - size
    base = min(int(total * args.start_fraction), max_start)
    if args.placement == "consecutive":
        if base + count * size > total:
            raise ValueError(
                "Consecutive windows exceed the available rows; reduce "
                "--n_windows/--flows_per_window or --start_fraction."
            )
        return [base + i * size for i in range(count)]

    # Spread non-overlapping windows over [base, total].  Equal spacing may be
    # larger than `size`, but never smaller because count*size <= remaining.
    if count == 1:
        return [base]
    remaining_max_start = total - size
    starts = [
        round(base + i * (remaining_max_start - base) / (count - 1))
        for i in range(count)
    ]
    if any(b - a < size for a, b in zip(starts, starts[1:])):
        raise ValueError(
            "The requested evenly-spread windows overlap after start_fraction; "
            "reduce --start_fraction or window sizes."
        )
    return starts


def main() -> None:
    args = parse_args()
    flow_path = Path(args.flow_csv)
    packet_path = Path(args.packet_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] reading and sorting flows: {flow_path}")
    flows = load_flows(flow_path)
    starts = choose_starts(len(flows), args)

    windows = []
    flow_to_window: dict[int, int] = {}
    for i, start in enumerate(starts, 1):
        selected = flows.iloc[start : start + args.flows_per_window].copy()
        name = f"{args.prefix}-w{i:02d}"
        flow_out = out_dir / f"{name}-stage1_flows.csv"
        packet_out = out_dir / f"{name}-stage1_packets.csv"
        selected.to_csv(flow_out, index=False)

        ids = selected["flow_id"].astype("int64").tolist()
        for flow_id in ids:
            if flow_id in flow_to_window:
                raise RuntimeError(f"overlapping flow_id between windows: {flow_id}")
            flow_to_window[flow_id] = i - 1

        labels = pd.to_numeric(selected["label"], errors="coerce").fillna(0)
        binary = (labels != 0).astype(int)
        windows.append(
            {
                "window": i,
                "name": name,
                "flow_csv": str(flow_out),
                "packet_csv": str(packet_out),
                "start_row": int(start),
                "flow_count": int(len(selected)),
                "benign_count": int((binary == 0).sum()),
                "malicious_count": int((binary == 1).sum()),
                "time_start_us": int(selected["flow_start_timestamp_us"].iloc[0]),
                "time_end_us": int(selected["flow_start_timestamp_us"].iloc[-1]),
                "packet_count": 0,
                "matched_flow_ids": set(),
            }
        )
        print(
            f"[WINDOW {i:02d}] flows={len(selected):,}, "
            f"benign={(binary == 0).sum():,}, malicious={(binary == 1).sum():,}"
        )

    wrote_header = [False] * len(windows)
    rows_seen = 0
    print(f"[INFO] scanning packets once: {packet_path}")
    for chunk_no, chunk in enumerate(
        pd.read_csv(packet_path, chunksize=args.chunksize, low_memory=False), 1
    ):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        if "flow_id" not in chunk.columns:
            raise ValueError("packet CSV missing column: flow_id")
        rows_seen += len(chunk)
        ids = pd.to_numeric(chunk["flow_id"], errors="coerce").astype("Int64")
        destinations = ids.map(flow_to_window)

        for window_idx in destinations.dropna().unique():
            idx = int(window_idx)
            sub = chunk.loc[destinations == idx].copy()
            if sub.empty:
                continue
            out = Path(windows[idx]["packet_csv"])
            sub.to_csv(
                out,
                index=False,
                mode="a" if wrote_header[idx] else "w",
                header=not wrote_header[idx],
            )
            wrote_header[idx] = True
            windows[idx]["packet_count"] += int(len(sub))
            matched = pd.to_numeric(
                sub["flow_id"], errors="coerce"
            ).dropna().astype("int64")
            windows[idx]["matched_flow_ids"].update(matched.unique().tolist())

        if chunk_no % args.progress_every_chunks == 0:
            kept = sum(w["packet_count"] for w in windows)
            print(
                f"[INFO] chunks={chunk_no}, rows_seen={rows_seen:,}, "
                f"packets_kept={kept:,}"
            )

    header = pd.read_csv(packet_path, nrows=0)
    for w in windows:
        if not Path(w["packet_csv"]).exists():
            header.to_csv(w["packet_csv"], index=False)
        w["matched_flow_count"] = len(w["matched_flow_ids"])
        w["missing_packet_flow_count"] = w["flow_count"] - w["matched_flow_count"]
        w["matched_flow_ids"] = sorted(w["matched_flow_ids"])

    report = {
        "source_flow_csv": str(flow_path),
        "source_packet_csv": str(packet_path),
        "placement": args.placement,
        "n_windows": args.n_windows,
        "flows_per_window": args.flows_per_window,
        "packet_rows_scanned": rows_seen,
        "windows": windows,
    }
    report_path = out_dir / f"{args.prefix}-windows-report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote {len(windows)} paired external-test sets to {out_dir}")
    print(f"[DONE] report: {report_path}")


if __name__ == "__main__":
    main()
