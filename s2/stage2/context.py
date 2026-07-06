from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List

import numpy as np
import pandas as pd

class ContextIndexBuilder:
    """Precompute context indices for every flow in chronological order.

    This avoids slow per-sample dataframe filtering in Dataset.__getitem__.
    The builder scans once through time and updates bounded histories.
    """

    def __init__(self, meta_df: pd.DataFrame, cfg: Dict[str, Any]):
        self.meta_df = meta_df.reset_index(drop=True)
        self.cfg = cfg

        self.source_col = cfg["data"].get("source_col", "source_id")
        self.destination_col = cfg["data"].get("destination_col", "destination_id")

        ctx = cfg["context"]
        self.method = ctx.get("method", "endpoint")
        self.window_size = int(ctx.get("window_size", 16))
        self.include_target = bool(ctx.get("include_target", True))

        if self.include_target:
            self.k = max(self.window_size - 1, 0)
        else:
            self.k = self.window_size

        self.context_policy = ctx.get("context_policy", "split_isolated")
        self.endpoint_mode = ctx.get("endpoint_mode", "same_endpoint")
        self.deduplicate = bool(ctx.get("deduplicate", True))
        print(f"[S2-ContextIndexBuilder]--context method: {self.method}, window_size: {self.window_size}, include_target: {self.include_target}, "
              f"context_policy: {self.context_policy}, endpoint_mode: {self.endpoint_mode}, deduplicate: {self.deduplicate}")
        self._validate()

    def _validate(self) -> None:
        if self.k < 0:
            raise ValueError("context.window_size must be >= 0")
        if self.method not in {"time_only", "source_host", "destination_host", "endpoint"}:
            raise ValueError(f"Unknown context.method: {self.method}")
        if self.context_policy not in {"online", "split_isolated", "train_only_for_eval"}:
            raise ValueError(f"Unknown context.context_policy: {self.context_policy}")
        if self.endpoint_mode not in {"same_endpoint", "same_source_or_dest"}:
            raise ValueError(f"Unknown context.endpoint_mode: {self.endpoint_mode}")

    def _allowed_context_split(self, target_split: str, candidate_split: str) -> bool:
        if self.context_policy == "online":
            return True

        if self.context_policy == "split_isolated":
            return target_split == candidate_split

        if self.context_policy == "train_only_for_eval":
            return candidate_split == "train"

        raise RuntimeError("unreachable")

    def _filter_by_policy(self, candidates: List[int], target_split: str, splits: List[str]) -> List[int]:
        if self.context_policy == "online":
            return candidates
        return [i for i in candidates if self._allowed_context_split(target_split, splits[i])]

    def _recent(self, candidates: List[int]) -> List[int]:
        if self.k == 0:
            return []

        if not self.deduplicate:
            return candidates[-self.k:]

        seen = set()
        out_reversed: List[int] = []
        for idx in reversed(candidates):
            if idx in seen:
                continue
            seen.add(idx)
            out_reversed.append(idx)
            if len(out_reversed) >= self.k:
                break

        return list(reversed(out_reversed))

    def build(self) -> List[np.ndarray]:
        n = len(self.meta_df)
        contexts: List[np.ndarray] = []

        sources = self.meta_df[self.source_col].astype(str).tolist()
        destinations = self.meta_df[self.destination_col].astype(str).tolist()
        splits = self.meta_df["split"].astype(str).tolist()

        # Larger than k because split policy filtering may remove many recent candidates.
        history_cap = max(self.k * 5, self.k + 32, 64)

        global_hist: deque[int] = deque(maxlen=history_cap)
        source_hist: Dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=history_cap))
        dest_hist: Dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=history_cap))
        endpoint_hist: Dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=history_cap))

        for idx in range(n):
            src = sources[idx]
            dst = destinations[idx]
            target_split = splits[idx]

            if self.method == "time_only":
                candidates = list(global_hist)

            elif self.method == "source_host":
                candidates = list(source_hist[src])

            elif self.method == "destination_host":
                candidates = list(dest_hist[dst])

            elif self.method == "endpoint":
                if self.endpoint_mode == "same_source_or_dest":
                    # candidates = list(source_hist[src]) + list(dest_hist[dst])
                    # 去重后排序
                    candidates = sorted(set(source_hist[src]).union(dest_hist[dst]))
                else:
                    # candidates = list(endpoint_hist[src]) + list(endpoint_hist[dst])
                    # 去重后排序
                    candidates = sorted(set(endpoint_hist[src]).union(endpoint_hist[dst]))

            else:
                raise RuntimeError("unreachable")

            candidates = self._filter_by_policy(candidates, target_split, splits)
            ctx = self._recent(candidates)

            # Append current flow after historical context. This does not leak label;
            # it only uses current Stage1 z_intra as the target token representation.
            if self.include_target:
                ctx.append(idx)

            contexts.append(np.array(ctx, dtype=np.int64))

            # Update histories only after current context is built.
            # This prevents self/future leakage.
            global_hist.append(idx)
            source_hist[src].append(idx)
            dest_hist[dst].append(idx)
            endpoint_hist[src].append(idx)
            if dst != src:
                endpoint_hist[dst].append(idx)
            # endpoint_hist[src].append(idx)
            # endpoint_hist[dst].append(idx)

        return contexts
