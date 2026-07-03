from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def read_columns(csv_path: str) -> List[str]:
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def require_columns(
    actual_columns: Iterable[str],
    required_columns: Iterable[str],
    file_label: str,
) -> None:
    actual = set(actual_columns)
    missing = [col for col in required_columns if col not in actual]
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}")


def save_sample(df: pd.DataFrame, path: str, sample_rows: int) -> str:
    if len(df) > sample_rows:
        df = df.head(sample_rows)
    df.to_csv(path, index=False)
    return path


def status_from_rate(
    bad_count: int,
    total_count: int,
    fail_tolerance: float,
    *,
    warn_tolerance: Optional[float] = None,
) -> str:
    rate = ratio(bad_count, total_count)
    if rate > fail_tolerance:
        return "FAIL"
    if warn_tolerance is not None and rate > warn_tolerance:
        return "WARN"
    return "PASS"


def add_check(
    checks: List[Dict[str, Any]],
    name: str,
    status: str,
    bad_count: int,
    total_count: int,
    message: str,
    *,
    threshold: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    item: Dict[str, Any] = {
        "name": name,
        "status": status,
        "bad_count": int(bad_count),
        "total_count": int(total_count),
        "bad_rate": ratio(int(bad_count), int(total_count)),
        "message": message,
    }
    if threshold is not None:
        item["threshold"] = float(threshold)
    if extra:
        item.update(extra)
    checks.append(item)


def aggregate_packet_csv(
    packet_csv: str,
    *,
    flow_id_col: str,
    chunksize: int,
) -> pd.DataFrame:
    packet_columns = read_columns(packet_csv)
    required = [flow_id_col]
    optional = [
        "timestamp_us",
        "direction",
        "pkt_len",
        "ip_len",
        "payload_len",
        "packet_label",
    ]
    require_columns(packet_columns, required, "packet CSV")
    usecols = [col for col in required + optional if col in packet_columns]

    partials: List[pd.DataFrame] = []

    for chunk_idx, chunk in enumerate(
        pd.read_csv(packet_csv, usecols=usecols, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        if chunk.empty:
            continue

        flow_ids = chunk[flow_id_col]
        grouped = chunk.groupby(flow_ids, sort=False)

        part = pd.DataFrame(index=grouped.size().index)
        part.index.name = flow_id_col
        part["packet_count_from_packet_csv"] = grouped.size().astype(np.int64)

        if "timestamp_us" in chunk.columns:
            ts = pd.to_numeric(chunk["timestamp_us"], errors="coerce")
            part["packet_ts_min"] = ts.groupby(flow_ids).min()
            part["packet_ts_max"] = ts.groupby(flow_ids).max()
            part["packet_ts_nan_count"] = ts.isna().groupby(flow_ids).sum().astype(np.int64)

        if "direction" in chunk.columns:
            direction = pd.to_numeric(chunk["direction"], errors="coerce")
            part["packet_fwd_count"] = (direction == 0).groupby(flow_ids).sum().astype(np.int64)
            part["packet_bwd_count"] = (direction == 1).groupby(flow_ids).sum().astype(np.int64)
            invalid_direction = ~direction.isin([0, 1])
            part["packet_invalid_direction_count"] = (
                invalid_direction.groupby(flow_ids).sum().astype(np.int64)
            )

        for col in ["pkt_len", "ip_len", "payload_len"]:
            if col in chunk.columns:
                values = pd.to_numeric(chunk[col], errors="coerce")
                part[f"{col}_sum_from_packets"] = values.fillna(0).groupby(flow_ids).sum()
                part[f"{col}_max_from_packets"] = values.groupby(flow_ids).max()
                part[f"{col}_nan_count"] = values.isna().groupby(flow_ids).sum().astype(np.int64)

        if "packet_label" in chunk.columns:
            packet_label = pd.to_numeric(chunk["packet_label"], errors="coerce")
            part["packet_label_max"] = packet_label.groupby(flow_ids).max()
            part["packet_label_nan_count"] = (
                packet_label.isna().groupby(flow_ids).sum().astype(np.int64)
            )

        partials.append(part.reset_index())
        print(f"[INFO] processed packet chunk {chunk_idx}: rows={len(chunk):,}")

    if not partials:
        raise ValueError(f"packet CSV is empty: {packet_csv}")

    all_parts = pd.concat(partials, ignore_index=True)
    grouped_parts = all_parts.groupby(flow_id_col, sort=False)

    agg_rules: Dict[str, str] = {
        "packet_count_from_packet_csv": "sum",
    }
    for col in all_parts.columns:
        if col == flow_id_col or col == "packet_count_from_packet_csv":
            continue
        if col.endswith("_min"):
            agg_rules[col] = "min"
        elif col.endswith("_max") or col == "packet_label_max":
            agg_rules[col] = "max"
        else:
            agg_rules[col] = "sum"

    packet_agg = grouped_parts.agg(agg_rules).reset_index()

    if {"packet_ts_min", "packet_ts_max"}.issubset(packet_agg.columns):
        packet_agg["packet_duration_us_from_packets"] = (
            packet_agg["packet_ts_max"] - packet_agg["packet_ts_min"]
        )

    for col in ["pkt_len", "ip_len", "payload_len"]:
        sum_col = f"{col}_sum_from_packets"
        if sum_col in packet_agg.columns:
            packet_agg[f"{col}_mean_from_packets"] = (
                packet_agg[sum_col] / packet_agg["packet_count_from_packet_csv"].clip(lower=1)
            )

    return packet_agg


def run_checks(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_dir(args.out_dir)

    flow_columns = read_columns(args.flow_csv)
    packet_columns = read_columns(args.packet_csv)

    require_columns(packet_columns, [args.flow_id_col], "packet CSV")
    require_columns(flow_columns, [args.flow_id_col], "flow CSV")

    flow_required_for_checks = [
        "total_fwd_packets",
        "total_backward_packets",
        "flow_duration",
        "label",
    ]
    missing_flow_check_cols = [
        col for col in flow_required_for_checks if col not in flow_columns
    ]
    if missing_flow_check_cols:
        raise ValueError(
            "flow CSV is missing columns required by these checks: "
            f"{missing_flow_check_cols}"
        )

    print("[INFO] loading flow CSV...")
    flows = pd.read_csv(args.flow_csv, low_memory=False)
    flows[args.flow_id_col] = pd.to_numeric(flows[args.flow_id_col], errors="coerce")

    if flows[args.flow_id_col].isna().any():
        bad = flows[flows[args.flow_id_col].isna()].copy()
        save_sample(
            bad,
            os.path.join(args.out_dir, "invalid_flow_id_rows_in_flow_csv.csv"),
            args.sample_rows,
        )
        flows = flows.dropna(subset=[args.flow_id_col]).copy()

    flows[args.flow_id_col] = flows[args.flow_id_col].astype(np.int64)

    print("[INFO] aggregating packet CSV by flow_id...")
    packet_agg = aggregate_packet_csv(
        args.packet_csv,
        flow_id_col=args.flow_id_col,
        chunksize=args.chunksize,
    )
    packet_agg[args.flow_id_col] = pd.to_numeric(
        packet_agg[args.flow_id_col], errors="coerce"
    )
    packet_agg = packet_agg.dropna(subset=[args.flow_id_col]).copy()
    packet_agg[args.flow_id_col] = packet_agg[args.flow_id_col].astype(np.int64)

    packet_agg_path = os.path.join(args.out_dir, "packet_flow_aggregate.csv")
    packet_agg.to_csv(packet_agg_path, index=False)

    checks: List[Dict[str, Any]] = []

    flow_rows = len(flows)
    unique_flow_rows = flows[args.flow_id_col].nunique()
    packet_unique_flows = packet_agg[args.flow_id_col].nunique()

    duplicate_flow_mask = flows[args.flow_id_col].duplicated(keep=False)
    duplicate_flow_rows = flows[duplicate_flow_mask].copy()
    duplicate_flow_ids = duplicate_flow_rows[args.flow_id_col].nunique()
    if duplicate_flow_ids > 0:
        save_sample(
            duplicate_flow_rows.sort_values(args.flow_id_col),
            os.path.join(args.out_dir, "duplicate_flow_ids_in_flow_csv.csv"),
            args.sample_rows,
        )

    add_check(
        checks,
        "duplicate flow_id in flow CSV",
        "FAIL" if duplicate_flow_ids > 0 else "PASS",
        duplicate_flow_ids,
        unique_flow_rows,
        "flow CSV should normally have one row per flow_id.",
        threshold=0.0,
        extra={"duplicate_flow_rows": int(len(duplicate_flow_rows))},
    )

    packet_ids = set(packet_agg[args.flow_id_col].tolist())
    flow_ids = set(flows[args.flow_id_col].tolist())

    packet_ids_missing_in_flow = sorted(packet_ids - flow_ids)
    flow_ids_missing_in_packet = sorted(flow_ids - packet_ids)

    packet_missing_df = packet_agg[
        packet_agg[args.flow_id_col].isin(packet_ids_missing_in_flow)
    ].copy()
    flow_missing_df = flows[flows[args.flow_id_col].isin(flow_ids_missing_in_packet)].copy()

    if len(packet_missing_df) > 0:
        save_sample(
            packet_missing_df,
            os.path.join(args.out_dir, "packet_flow_ids_missing_in_flow_csv.csv"),
            args.sample_rows,
        )

    if len(flow_missing_df) > 0:
        save_sample(
            flow_missing_df,
            os.path.join(args.out_dir, "flow_ids_missing_in_packet_csv.csv"),
            args.sample_rows,
        )

    add_check(
        checks,
        "packet flow_id missing in flow CSV",
        status_from_rate(
            len(packet_ids_missing_in_flow),
            packet_unique_flows,
            args.missing_flow_id_tolerance,
        ),
        len(packet_ids_missing_in_flow),
        packet_unique_flows,
        "Every packet flow_id should normally have a matching row in the flow CSV.",
        threshold=args.missing_flow_id_tolerance,
    )

    add_check(
        checks,
        "flow CSV flow_id missing in packet CSV",
        status_from_rate(
            len(flow_ids_missing_in_packet),
            unique_flow_rows,
            args.missing_flow_id_tolerance,
        ),
        len(flow_ids_missing_in_packet),
        unique_flow_rows,
        "Every flow CSV row should normally have at least one packet in packet CSV.",
        threshold=args.missing_flow_id_tolerance,
    )

    merged = flows.merge(
        packet_agg,
        on=args.flow_id_col,
        how="outer",
        indicator=True,
        suffixes=("_flow", "_packet"),
    )

    for col in [
        "total_fwd_packets",
        "total_backward_packets",
        "flow_duration",
        "label",
        "packet_count_from_packet_csv",
        "packet_fwd_count",
        "packet_bwd_count",
        "packet_invalid_direction_count",
        "packet_label_max",
        "pkt_len_mean_from_packets",
        "ip_len_mean_from_packets",
        "payload_len_mean_from_packets",
    ]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    matched = merged[merged["_merge"] == "both"].copy()
    matched_count = len(matched)

    matched["flow_total_packets"] = (
        matched["total_fwd_packets"].fillna(0)
        + matched["total_backward_packets"].fillna(0)
    )

    count_mismatch = matched[
        matched["packet_count_from_packet_csv"].fillna(-1).astype(np.int64)
        != matched["flow_total_packets"].fillna(-2).astype(np.int64)
    ].copy()
    if len(count_mismatch) > 0:
        save_sample(
            count_mismatch[
                [
                    args.flow_id_col,
                    "packet_count_from_packet_csv",
                    "total_fwd_packets",
                    "total_backward_packets",
                    "flow_total_packets",
                ]
            ],
            os.path.join(args.out_dir, "packet_count_mismatch.csv"),
            args.sample_rows,
        )

    add_check(
        checks,
        "packet count mismatch",
        status_from_rate(
            len(count_mismatch),
            matched_count,
            args.packet_count_mismatch_tolerance,
        ),
        len(count_mismatch),
        matched_count,
        "packet CSV group size should equal total_fwd_packets + total_backward_packets.",
        threshold=args.packet_count_mismatch_tolerance,
    )

    if {"packet_fwd_count", "packet_bwd_count"}.issubset(matched.columns):
        direction_count_mismatch = matched[
            (matched["packet_fwd_count"].fillna(-1).astype(np.int64)
             != matched["total_fwd_packets"].fillna(-2).astype(np.int64))
            | (matched["packet_bwd_count"].fillna(-1).astype(np.int64)
               != matched["total_backward_packets"].fillna(-2).astype(np.int64))
        ].copy()
        if len(direction_count_mismatch) > 0:
            save_sample(
                direction_count_mismatch[
                    [
                        args.flow_id_col,
                        "packet_fwd_count",
                        "total_fwd_packets",
                        "packet_bwd_count",
                        "total_backward_packets",
                    ]
                ],
                os.path.join(args.out_dir, "direction_count_mismatch.csv"),
                args.sample_rows,
            )

        add_check(
            checks,
            "direction count mismatch",
            status_from_rate(
                len(direction_count_mismatch),
                matched_count,
                args.packet_count_mismatch_tolerance,
            ),
            len(direction_count_mismatch),
            matched_count,
            "Packet direction counts should match total_fwd_packets and total_backward_packets.",
            threshold=args.packet_count_mismatch_tolerance,
        )

    if "packet_invalid_direction_count" in matched.columns:
        invalid_direction_flows = matched[
            matched["packet_invalid_direction_count"].fillna(0) > 0
        ].copy()
        if len(invalid_direction_flows) > 0:
            save_sample(
                invalid_direction_flows[
                    [
                        args.flow_id_col,
                        "packet_invalid_direction_count",
                        "packet_count_from_packet_csv",
                    ]
                ],
                os.path.join(args.out_dir, "invalid_packet_direction_flows.csv"),
                args.sample_rows,
            )

        add_check(
            checks,
            "invalid packet direction values",
            "FAIL" if len(invalid_direction_flows) > 0 else "PASS",
            len(invalid_direction_flows),
            matched_count,
            "Packet direction should be 0 or 1 for all packets used in Stage1.",
            threshold=0.0,
        )

    multi_packet = matched[matched["flow_total_packets"] > 1].copy()
    zero_duration_multi = multi_packet[multi_packet["flow_duration"].fillna(0) == 0].copy()
    if len(zero_duration_multi) > 0:
        cols = [
            args.flow_id_col,
            "flow_duration",
            "flow_total_packets",
            "packet_count_from_packet_csv",
        ]
        if "packet_duration_us_from_packets" in zero_duration_multi.columns:
            cols.append("packet_duration_us_from_packets")
        save_sample(
            zero_duration_multi[cols],
            os.path.join(args.out_dir, "zero_duration_multi_packet_flows.csv"),
            args.sample_rows,
        )

    add_check(
        checks,
        "multi-packet flow_duration == 0",
        status_from_rate(
            len(zero_duration_multi),
            len(multi_packet),
            args.zero_duration_multi_tolerance,
        ),
        len(zero_duration_multi),
        len(multi_packet),
        "Single-packet duration=0 is fine; multi-packet duration=0 is suspicious.",
        threshold=args.zero_duration_multi_tolerance,
    )

    if "packet_duration_us_from_packets" in matched.columns:
        duration_disagree = matched[
            (matched["flow_total_packets"] > 1)
            & (matched["packet_duration_us_from_packets"].fillna(0) > 0)
            & (matched["flow_duration"].fillna(0) == 0)
        ].copy()
        if len(duration_disagree) > 0:
            save_sample(
                duration_disagree[
                    [
                        args.flow_id_col,
                        "flow_duration",
                        "packet_duration_us_from_packets",
                        "flow_total_packets",
                    ]
                ],
                os.path.join(args.out_dir, "duration_zero_but_packet_timestamps_positive.csv"),
                args.sample_rows,
            )

        add_check(
            checks,
            "flow duration disagrees with packet timestamps",
            status_from_rate(
                len(duration_disagree),
                len(multi_packet),
                args.zero_duration_multi_tolerance,
            ),
            len(duration_disagree),
            len(multi_packet),
            "If packet timestamp range is positive but flow_duration is zero, aggregation timing may be wrong.",
            threshold=args.zero_duration_multi_tolerance,
        )

    zero_bwd = matched[matched["total_backward_packets"].fillna(0) == 0].copy()
    zero_bwd_multi = multi_packet[multi_packet["total_backward_packets"].fillna(0) == 0].copy()
    if len(zero_bwd_multi) > 0:
        save_sample(
            zero_bwd_multi[
                [
                    args.flow_id_col,
                    "total_fwd_packets",
                    "total_backward_packets",
                    "flow_total_packets",
                    "protocol",
                ]
                if "protocol" in zero_bwd_multi.columns
                else [
                    args.flow_id_col,
                    "total_fwd_packets",
                    "total_backward_packets",
                    "flow_total_packets",
                ]
            ],
            os.path.join(args.out_dir, "zero_backward_multi_packet_flows.csv"),
            args.sample_rows,
        )

    zero_bwd_rate = ratio(len(zero_bwd), matched_count)
    direction_status = "WARN" if zero_bwd_rate >= args.zero_backward_warn_ratio else "PASS"
    add_check(
        checks,
        "flows with total_backward_packets == 0",
        direction_status,
        len(zero_bwd),
        matched_count,
        "A high one-way-flow ratio may be normal for some PCAPs, but is suspicious for mostly bidirectional traffic.",
        threshold=args.zero_backward_warn_ratio,
        extra={
            "multi_packet_zero_backward_count": int(len(zero_bwd_multi)),
            "multi_packet_zero_backward_rate": ratio(len(zero_bwd_multi), len(multi_packet)),
        },
    )

    length_cols_available = all(
        col in matched.columns
        for col in [
            "packet_length_mean",
            "total_length_of_fwd_packets",
            "total_length_of_bwd_packets",
        ]
    )
    if length_cols_available:
        flow_len_zero = (
            (matched["packet_length_mean"].fillna(0) == 0)
            & (matched["total_length_of_fwd_packets"].fillna(0) == 0)
            & (matched["total_length_of_bwd_packets"].fillna(0) == 0)
        )

        packet_len_positive_terms = []
        for col in ["pkt_len_mean_from_packets", "ip_len_mean_from_packets"]:
            if col in matched.columns:
                packet_len_positive_terms.append(matched[col].fillna(0) > 0)
        if packet_len_positive_terms:
            packet_len_positive = packet_len_positive_terms[0]
            for term in packet_len_positive_terms[1:]:
                packet_len_positive = packet_len_positive | term
        else:
            packet_len_positive = pd.Series(False, index=matched.index)

        length_suspicious = matched[flow_len_zero & packet_len_positive].copy()
        if len(length_suspicious) > 0:
            cols = [
                args.flow_id_col,
                "packet_length_mean",
                "total_length_of_fwd_packets",
                "total_length_of_bwd_packets",
                "pkt_len_mean_from_packets",
                "ip_len_mean_from_packets",
                "payload_len_mean_from_packets",
                "flow_total_packets",
            ]
            cols = [col for col in cols if col in length_suspicious.columns]
            save_sample(
                length_suspicious[cols],
                os.path.join(args.out_dir, "flow_length_zero_but_packet_len_positive.csv"),
                args.sample_rows,
            )

        length_rate = ratio(len(length_suspicious), matched_count)
        length_status = "WARN" if length_rate >= args.length_zero_warn_ratio else "PASS"
        add_check(
            checks,
            "flow length stats zero but packet length positive",
            length_status,
            len(length_suspicious),
            matched_count,
            "This often means flow length stats are payload-based; consider enhanced IP-length flow features.",
            threshold=args.length_zero_warn_ratio,
        )

    if "packet_label_max" in matched.columns:
        label_mismatch = matched[
            matched["label"].fillna(-1).astype(np.int64)
            != matched["packet_label_max"].fillna(-2).astype(np.int64)
        ].copy()
        if len(label_mismatch) > 0:
            save_sample(
                label_mismatch[
                    [
                        args.flow_id_col,
                        "label",
                        "packet_label_max",
                        "packet_count_from_packet_csv",
                    ]
                ],
                os.path.join(args.out_dir, "label_mismatch_flow_vs_packet_max.csv"),
                args.sample_rows,
            )

        add_check(
            checks,
            "flow label != max(packet_label)",
            status_from_rate(
                len(label_mismatch),
                matched_count,
                args.label_mismatch_tolerance,
            ),
            len(label_mismatch),
            matched_count,
            "Flow label should equal max packet_label if your plugin defines flow label as any alert in the flow.",
            threshold=args.label_mismatch_tolerance,
        )

    merged_path = os.path.join(args.out_dir, "flow_packet_consistency_join.csv")
    merged.to_csv(merged_path, index=False)

    summary: Dict[str, Any] = {
        "packet_csv": os.path.abspath(args.packet_csv),
        "flow_csv": os.path.abspath(args.flow_csv),
        "out_dir": os.path.abspath(args.out_dir),
        "counts": {
            "flow_csv_rows": int(flow_rows),
            "flow_csv_unique_flow_ids": int(unique_flow_rows),
            "packet_csv_unique_flow_ids": int(packet_unique_flows),
            "matched_flow_ids": int(matched_count),
            "packet_flow_ids_missing_in_flow_csv": int(len(packet_ids_missing_in_flow)),
            "flow_ids_missing_in_packet_csv": int(len(flow_ids_missing_in_packet)),
        },
        "checks": checks,
        "artifacts": {
            "packet_flow_aggregate_csv": os.path.abspath(packet_agg_path),
            "flow_packet_consistency_join_csv": os.path.abspath(merged_path),
        },
    }

    summary_path = os.path.join(args.out_dir, "suricata_feature_checks_summary.json")
    save_json(summary, summary_path)

    report_path = os.path.join(args.out_dir, "suricata_feature_checks_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Suricata CSV Consistency Checks\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"packet_csv: {summary['packet_csv']}\n")
        f.write(f"flow_csv  : {summary['flow_csv']}\n")
        f.write(f"out_dir   : {summary['out_dir']}\n\n")
        f.write("Counts\n")
        f.write("-" * 40 + "\n")
        for key, value in summary["counts"].items():
            f.write(f"{key}: {value}\n")
        f.write("\nChecks\n")
        f.write("-" * 40 + "\n")
        for check in checks:
            f.write(
                f"[{check['status']}] {check['name']}: "
                f"{check['bad_count']}/{check['total_count']} "
                f"({check['bad_rate']:.6f})\n"
            )
            f.write(f"  {check['message']}\n")
            if "threshold" in check:
                f.write(f"  threshold: {check['threshold']}\n")
            f.write("\n")

    print("\n[SUMMARY]")
    for check in checks:
        print(
            f"[{check['status']}] {check['name']}: "
            f"{check['bad_count']}/{check['total_count']} "
            f"({check['bad_rate']:.6f})"
        )
    print(f"\n[INFO] saved summary: {summary_path}")
    print(f"[INFO] saved report : {report_path}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity-check Suricata packet CSV and flow CSV consistency."
    )
    parser.add_argument("--packet_csv", required=True)
    parser.add_argument("--flow_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--flow_id_col", default="flow_id")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--sample_rows", type=int, default=1000)

    parser.add_argument("--missing_flow_id_tolerance", type=float, default=0.001)
    parser.add_argument("--packet_count_mismatch_tolerance", type=float, default=0.001)
    parser.add_argument("--zero_duration_multi_tolerance", type=float, default=0.001)
    parser.add_argument("--label_mismatch_tolerance", type=float, default=0.001)
    parser.add_argument("--zero_backward_warn_ratio", type=float, default=0.95)
    parser.add_argument("--length_zero_warn_ratio", type=float, default=0.50)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_checks(args)


if __name__ == "__main__":
    main()
