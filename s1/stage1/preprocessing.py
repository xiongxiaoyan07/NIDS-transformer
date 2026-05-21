"""
Preprocessing:
- configurable feature columns
- one-hot encoding for categorical columns
- log1p transform for selected heavy-tailed numerical columns
- quantile clipping / winsorization for numerical columns
- StandardScaler / RobustScaler / none
- fit only on training data to avoid leakage
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler


def make_onehot_encoder() -> OneHotEncoder:
    """
    Support both old and new sklearn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_numeric_scaler(name: str):
    """
    Build numerical scaler.

    standard:
        mean/std normalization.
        Suitable after log1p + clipping.

    robust:
        median/IQR normalization.
        More robust to outliers.

    none:
        no scaling.
    """
    name = str(name).lower()

    if name == "standard":
        return StandardScaler()

    if name == "robust":
        return RobustScaler()

    if name == "none":
        return None

    raise ValueError(f"Unsupported numerical scaler: {name}")


class Stage1Preprocessor:
    """
    Fit/transform object for packet-level and flow-level features.

    Fit only on train data:
        packet_train_df, flow_train_df

    Transform:
        packet dataframe -> packet feature matrix
        flow dataframe   -> flow feature matrix

    Numerical pipeline:
        raw numeric values
        -> selected log1p transform
        -> train-fitted quantile clipping
        -> scaler

    Important:
        clipping bounds and scalers are fitted only on train data.
        validation/test/external-test reuse the same bounds and scalers.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
    ):
        self.cfg = cfg
        self.strict_schema = bool(cfg.get("strict_schema", False))

        data_cfg = cfg.get("data", {})
        self.packet_iat_col = data_cfg.get("packet_iat_col")

        feature_cfg = cfg.get("features", {})
        packet_cfg = feature_cfg.get("packet", {})
        flow_cfg = feature_cfg.get("flow", {})

        self.packet_num_requested = list(packet_cfg.get("numerical", []))
        self.packet_cat_requested = list(packet_cfg.get("categorical", []))
        self.packet_bin_requested = list(packet_cfg.get("binary", []))

        self.flow_num_requested = list(flow_cfg.get("numerical", []))
        self.flow_cat_requested = list(flow_cfg.get("categorical", []))
        self.flow_bin_requested = list(flow_cfg.get("binary", []))

        preprocessing_cfg = cfg.get("preprocessing", {})
        numerical_cfg = preprocessing_cfg.get("numerical", {})

        self.scaler_name = numerical_cfg.get("scaler", "standard")

        clip_cfg = numerical_cfg.get("clip", {})
        self.clip_enabled = bool(clip_cfg.get("enabled", True))
        self.clip_lower_quantile = float(clip_cfg.get("lower_quantile", 0.001))
        self.clip_upper_quantile = float(clip_cfg.get("upper_quantile", 0.999))

        if not 0.0 <= self.clip_lower_quantile <= 1.0:
            raise ValueError("clip.lower_quantile must be between 0 and 1.")
        if not 0.0 <= self.clip_upper_quantile <= 1.0:
            raise ValueError("clip.upper_quantile must be between 0 and 1.")
        if self.clip_lower_quantile > self.clip_upper_quantile:
            raise ValueError("clip.lower_quantile cannot be larger than upper_quantile.")

        log1p_cfg = numerical_cfg.get("log1p", {})
        self.packet_log1p_requested = set(log1p_cfg.get("packet", []))
        self.flow_log1p_requested = set(log1p_cfg.get("flow", []))

        self.packet_num_cols: List[str] = []
        self.packet_cat_cols: List[str] = []
        self.packet_bin_cols: List[str] = []

        self.flow_num_cols: List[str] = []
        self.flow_cat_cols: List[str] = []
        self.flow_bin_cols: List[str] = []

        self.packet_scaler = make_numeric_scaler(self.scaler_name)
        self.flow_scaler = make_numeric_scaler(self.scaler_name)

        self.packet_ohe = make_onehot_encoder()
        self.flow_ohe = make_onehot_encoder()

        # clipping bounds are learned from train data only
        # format:
        #   {"column_name": (lower_bound, upper_bound)}
        self.packet_clip_bounds: Dict[str, Tuple[float, float]] = {}
        self.flow_clip_bounds: Dict[str, Tuple[float, float]] = {}

        self.fitted = False

    def add_derived_packet_features(self, packets: pd.DataFrame) -> pd.DataFrame:
        """
        Add log_flow_iat_us.

        The raw flow_iat_us can have a heavy-tailed distribution.
        log(1 + x) makes the temporal feature more stable.

        Note:
            The raw flow_iat_us can still be used by the time-aware positional encoding.
            log_flow_iat_us is used as a normal numerical packet feature.
        """
        df = packets.copy()

        if self.packet_iat_col not in df.columns:
            df[self.packet_iat_col] = 0

        df[self.packet_iat_col] = pd.to_numeric(
            df[self.packet_iat_col], errors="coerce"
        ).fillna(0).clip(lower=0)

        df["log_flow_iat_us"] = np.log1p(
            df[self.packet_iat_col].astype(np.float64)
        ).astype(np.float32)

        return df

    def fit(self, packet_train: pd.DataFrame, flow_train: pd.DataFrame) -> None:
        """
        Fit scalers, encoders, and clipping bounds on train split only.
        """
        packet_train = self.add_derived_packet_features(packet_train)

        self.packet_num_cols = self._resolve_columns(
            packet_train, self.packet_num_requested, "packet numerical"
        )
        self.packet_cat_cols = self._resolve_columns(
            packet_train, self.packet_cat_requested, "packet categorical"
        )
        self.packet_bin_cols = self._resolve_columns(
            packet_train, self.packet_bin_requested, "packet binary"
        )

        self.flow_num_cols = self._resolve_columns(
            flow_train, self.flow_num_requested, "flow numerical"
        )
        self.flow_cat_cols = self._resolve_columns(
            flow_train, self.flow_cat_requested, "flow categorical"
        )
        self.flow_bin_cols = self._resolve_columns(
            flow_train, self.flow_bin_requested, "flow binary"
        )

        if not self.packet_num_cols and not self.packet_cat_cols and not self.packet_bin_cols:
            raise ValueError("No packet features selected.")

        if not self.flow_num_cols and not self.flow_cat_cols and not self.flow_bin_cols:
            print("[WARNING] No flow features selected. Flow features will not be used.")
            # Don't raise error, just skip flow feature fitting 可以没有flow特征，只使用packet的
        else:
            flow_num = self._numeric_matrix(
                flow_train,
                self.flow_num_cols,
                scope="flow",
                fit_clip=True,
            )

            if flow_num.shape[1] > 0 and self.flow_scaler is not None:
                self.flow_scaler.fit(flow_num)

            flow_cat = self._categorical_matrix(flow_train, self.flow_cat_cols)

            if flow_cat.shape[1] > 0:
                self.flow_ohe.fit(flow_cat)

        packet_num = self._numeric_matrix(
            packet_train,
            self.packet_num_cols,
            scope="packet",
            fit_clip=True,
        )

        if packet_num.shape[1] > 0 and self.packet_scaler is not None:
            self.packet_scaler.fit(packet_num)

        packet_cat = self._categorical_matrix(packet_train, self.packet_cat_cols)

        if packet_cat.shape[1] > 0:
            self.packet_ohe.fit(packet_cat)

        self.fitted = True

    def transform_packets_only(self, packets: pd.DataFrame) -> np.ndarray:
        """
        Transform only packet features (without flow features concatenated).
        Used for hierarchical fusion mode.

        Output order:
            [scaled numerical features;
             one-hot categorical features;
             raw binary features]
        """
        self._check_fitted()
        packets = self.add_derived_packet_features(packets)

        parts = []

        packet_num = self._numeric_matrix(
            packets,
            self.packet_num_cols,
            scope="packet",
            fit_clip=False,
        )

        if packet_num.shape[1] > 0:
            packet_num = self._scale_numeric(packet_num, self.packet_scaler)
            parts.append(packet_num)

        packet_cat = self._categorical_matrix(packets, self.packet_cat_cols)

        if packet_cat.shape[1] > 0:
            parts.append(self.packet_ohe.transform(packet_cat).astype(np.float32))

        packet_bin = self._binary_matrix(packets, self.packet_bin_cols)

        if packet_bin.shape[1] > 0:
            parts.append(packet_bin)

        if not parts:
            return np.zeros((len(packets), 0), dtype=np.float32)

        return np.concatenate(parts, axis=1).astype(np.float32)

    def transform_packets(self, packets: pd.DataFrame) -> np.ndarray:
        """
        Transform packet rows into model-ready features.

        Output order:
            [scaled numerical features;
             one-hot categorical features;
             raw binary features]
        """
        self._check_fitted()
        packets = self.add_derived_packet_features(packets)

        parts = []

        packet_num = self._numeric_matrix(
            packets,
            self.packet_num_cols,
            scope="packet",
            fit_clip=False,
        )

        if packet_num.shape[1] > 0:
            packet_num = self._scale_numeric(packet_num, self.packet_scaler)
            parts.append(packet_num)

        packet_cat = self._categorical_matrix(packets, self.packet_cat_cols)

        if packet_cat.shape[1] > 0:
            parts.append(self.packet_ohe.transform(packet_cat).astype(np.float32))

        packet_bin = self._binary_matrix(packets, self.packet_bin_cols)

        if packet_bin.shape[1] > 0:
            parts.append(packet_bin)

        if not parts:
            return np.zeros((len(packets), 0), dtype=np.float32)

        return np.concatenate(parts, axis=1).astype(np.float32)

    def transform_flows(self, flows: pd.DataFrame) -> np.ndarray:
        """
        Transform flow rows into model-ready features.

        Output order:
            [scaled numerical features;
             one-hot categorical features;
             raw binary features]
        """
        self._check_fitted()

        parts = []

        flow_num = self._numeric_matrix(
            flows,
            self.flow_num_cols,
            scope="flow",
            fit_clip=False,
        )

        if flow_num.shape[1] > 0:
            flow_num = self._scale_numeric(flow_num, self.flow_scaler)
            parts.append(flow_num)

        flow_cat = self._categorical_matrix(flows, self.flow_cat_cols)

        if flow_cat.shape[1] > 0:
            parts.append(self.flow_ohe.transform(flow_cat).astype(np.float32))

        flow_bin = self._binary_matrix(flows, self.flow_bin_cols)

        if flow_bin.shape[1] > 0:
            parts.append(flow_bin)

        if not parts:
            return np.zeros((len(flows), 0), dtype=np.float32)

        return np.concatenate(parts, axis=1).astype(np.float32)

    def input_dim(self) -> int:
        """
        Final x_i,t dimension:
            packet_feature_dim + flow_feature_dim (when inject_to_packets=True)
            or packet_feature_dim only (when inject_to_packets=False)
        """
        # Check the config to determine behavior
        flow_fusion_cfg = self.cfg.get("features", {}).get("flow_fusion", {})
        inject_to_packets = flow_fusion_cfg.get("inject_to_packets", True)

        if inject_to_packets:
            return self.packet_feature_dim() + self.flow_feature_dim()
        else:
            return self.packet_feature_dim()

    def packet_feature_dim(self) -> int:
        dim = len(self.packet_num_cols)

        if self.packet_cat_cols:
            dim += len(self.packet_ohe.get_feature_names_out())

        dim += len(self.packet_bin_cols)

        return int(dim)

    def flow_feature_dim(self) -> int:
        dim = len(self.flow_num_cols)

        if self.flow_cat_cols:
            dim += len(self.flow_ohe.get_feature_names_out())

        dim += len(self.flow_bin_cols)

        return int(dim)

    def has_flow_features(self) -> bool:
        """Check if any flow features are configured."""
        return (len(self.flow_num_cols) > 0 or
                len(self.flow_cat_cols) > 0 or
                len(self.flow_bin_cols) > 0)

    def summary(self) -> Dict[str, Any]:
        """
        Summary for debugging and reproducibility.
        """
        packet_log1p_used = [
            c for c in self.packet_num_cols if c in self.packet_log1p_requested
        ]
        flow_log1p_used = [
            c for c in self.flow_num_cols if c in self.flow_log1p_requested
        ]

        return {
            "packet_numerical_used": self.packet_num_cols,
            "packet_categorical_used": self.packet_cat_cols,
            "packet_binary_used": self.packet_bin_cols,

            "flow_numerical_used": self.flow_num_cols,
            "flow_categorical_used": self.flow_cat_cols,
            "flow_binary_used": self.flow_bin_cols,

            "packet_log1p_used": packet_log1p_used,
            "flow_log1p_used": flow_log1p_used,

            "scaler": self.scaler_name,
            "clip_enabled": self.clip_enabled,
            "clip_lower_quantile": self.clip_lower_quantile,
            "clip_upper_quantile": self.clip_upper_quantile,

            "packet_clip_bounds_count": len(self.packet_clip_bounds),
            "flow_clip_bounds_count": len(self.flow_clip_bounds),

            "packet_feature_dim": self.packet_feature_dim(),
            "flow_feature_dim": self.flow_feature_dim(),
            "input_dim": self.input_dim(),
            "has_flow_features": self.has_flow_features(),
        }

    def _resolve_columns(
        self,
        df: pd.DataFrame,
        requested: List[str],
        group_name: str,
    ) -> List[str]:
        """
        Check whether configured columns exist.

        If strict_schema=True:
            missing columns raise error.

        If strict_schema=False:
            missing columns are ignored.
        """
        present = [c for c in requested if c in df.columns]
        missing = [c for c in requested if c not in df.columns]

        if missing and self.strict_schema:
            raise ValueError(f"Missing configured {group_name} columns: {missing}")

        return present

    def _numeric_matrix(
        self,
        df: pd.DataFrame,
        cols: List[str],
        scope: str,
        fit_clip: bool,
    ) -> np.ndarray:
        """
        Build numerical feature matrix.

        Pipeline:
            1. numeric conversion
            2. replace inf with NaN
            3. selected log1p transform
            4. fill NaN
            5. train-fitted quantile clipping
            6. output float32 matrix

        scope:
            "packet" or "flow"
        """
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)

        x = df[cols].apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)

        x = self._apply_log1p(x, scope=scope)

        x = x.fillna(0.0)

        if self.clip_enabled:
            if fit_clip:
                self._fit_clip_bounds(x, scope=scope)

            x = self._apply_clip_bounds(x, scope=scope)

        return x.astype(np.float32).to_numpy()

    def _apply_log1p(self, x: pd.DataFrame, scope: str) -> pd.DataFrame:
        """
        Apply log1p to selected non-negative heavy-tailed columns.

        Important:
            Only columns listed in YAML will be transformed.
            Negative values are clipped to zero before log1p.
            This is suitable for counts, bytes, durations, rates, and IAT values.

        Do not include fields such as:
            src_port, dst_port, ttl_or_hop_limit, tcp_seq, tcp_ack
        unless you intentionally want this behavior.
        """
        x = x.copy()

        if scope == "packet":
            log_cols = self.packet_log1p_requested
        elif scope == "flow":
            log_cols = self.flow_log1p_requested
        else:
            raise ValueError(f"Unknown scope: {scope}")

        for col in x.columns:
            if col not in log_cols:
                continue

            values = pd.to_numeric(x[col], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            values = values.clip(lower=0.0)

            x[col] = np.log1p(values.astype(np.float64))

        return x

    def _fit_clip_bounds(self, x: pd.DataFrame, scope: str) -> None:
        """
        Fit clipping bounds from train data only.
        """
        bounds = self._get_clip_bounds_dict(scope)
        bounds.clear()

        for col in x.columns:
            values = pd.to_numeric(x[col], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()

            if len(values) == 0:
                bounds[col] = (0.0, 0.0)
                continue

            lower = float(values.quantile(self.clip_lower_quantile))
            upper = float(values.quantile(self.clip_upper_quantile))

            if not np.isfinite(lower):
                lower = 0.0

            if not np.isfinite(upper):
                upper = 0.0

            if lower > upper:
                lower, upper = upper, lower

            bounds[col] = (lower, upper)

    def _apply_clip_bounds(self, x: pd.DataFrame, scope: str) -> pd.DataFrame:
        """
        Apply train-fitted clipping bounds to train/val/test/external-test.
        """
        x = x.copy()
        bounds = self._get_clip_bounds_dict(scope)

        for col in x.columns:
            if col not in bounds:
                continue

            lower, upper = bounds[col]
            x[col] = x[col].clip(lower=lower, upper=upper)

        return x

    def _get_clip_bounds_dict(self, scope: str) -> Dict[str, Tuple[float, float]]:
        if scope == "packet":
            return self.packet_clip_bounds

        if scope == "flow":
            return self.flow_clip_bounds

        raise ValueError(f"Unknown scope: {scope}")

    @staticmethod
    def _scale_numeric(x: np.ndarray, scaler) -> np.ndarray:
        """
        Apply fitted scaler.

        If scaler is None, return the matrix unchanged.
        """
        if scaler is None:
            return x.astype(np.float32)

        return scaler.transform(x).astype(np.float32)

    @staticmethod
    def _categorical_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        if not cols:
            return np.zeros((len(df), 0), dtype=object)

        x = df[cols].copy()
        x = x.fillna("MISSING").astype(str)
        return x.to_numpy()

    @staticmethod
    def _binary_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        """
        Convert 0/1 binary columns to float32.

        Rules:
            NaN -> 0
            value > 0 -> 1
            value <= 0 -> 0

        No one-hot.
        No log1p.
        No clipping.
        No scaling.
        """
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)

        x = df[cols].apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        x = (x > 0).astype(np.float32)

        return x.to_numpy(dtype=np.float32)

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Stage1Preprocessor is not fitted.")