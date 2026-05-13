"""
Preprocessing:
- configurable feature columns
- one-hot encoding for categorical columns
- StandardScaler normalization for numerical columns
- fit only on training data to avoid leakage
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_onehot_encoder() -> OneHotEncoder:
    """
    Support both old and new sklearn versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class Stage1Preprocessor:
    """
    Fit/transform object for packet-level and flow-level features.

    Fit only on train data:
        packet_train_df, flow_train_df

    Transform:
        packet dataframe -> packet feature matrix
        flow dataframe   -> flow feature matrix
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
    ):
        self.cfg = cfg
        self.strict_schema = bool(cfg.get("strict_schema", False))

        data_cfg = cfg.get("data", {})
        self.packet_iat_col = data_cfg.get("packet_iat_col", "flow_iat_us")

        feature_cfg = cfg.get("features", {})
        packet_cfg = feature_cfg.get("packet", {})
        flow_cfg = feature_cfg.get("flow", {})

        self.packet_num_requested = list(packet_cfg.get("numerical", []))
        self.packet_cat_requested = list(packet_cfg.get("categorical", []))
        self.flow_num_requested = list(flow_cfg.get("numerical", []))
        self.flow_cat_requested = list(flow_cfg.get("categorical", []))

        self.packet_num_cols: List[str] = []
        self.packet_cat_cols: List[str] = []
        self.flow_num_cols: List[str] = []
        self.flow_cat_cols: List[str] = []

        self.packet_scaler = StandardScaler()
        self.flow_scaler = StandardScaler()
        self.packet_ohe = make_onehot_encoder()
        self.flow_ohe = make_onehot_encoder()

        self.fitted = False

    def add_derived_packet_features(self, packets: pd.DataFrame) -> pd.DataFrame:
        """
        Add log_flow_iat_us.

        The raw flow_iat_us can have a heavy-tailed distribution.
        log(1 + x) makes the temporal feature more stable.
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
        Fit scalers and encoders on train split only.
        """
        packet_train = self.add_derived_packet_features(packet_train)

        self.packet_num_cols = self._resolve_columns(
            packet_train, self.packet_num_requested, "packet numerical"
        )
        self.packet_cat_cols = self._resolve_columns(
            packet_train, self.packet_cat_requested, "packet categorical"
        )
        self.flow_num_cols = self._resolve_columns(
            flow_train, self.flow_num_requested, "flow numerical"
        )
        self.flow_cat_cols = self._resolve_columns(
            flow_train, self.flow_cat_requested, "flow categorical"
        )

        if not self.packet_num_cols and not self.packet_cat_cols:
            raise ValueError("No packet features selected.")
        if not self.flow_num_cols and not self.flow_cat_cols:
            raise ValueError("No flow features selected.")

        packet_num = self._numeric_matrix(packet_train, self.packet_num_cols)
        flow_num = self._numeric_matrix(flow_train, self.flow_num_cols)

        if packet_num.shape[1] > 0:
            self.packet_scaler.fit(packet_num)
        if flow_num.shape[1] > 0:
            self.flow_scaler.fit(flow_num)

        packet_cat = self._categorical_matrix(packet_train, self.packet_cat_cols)
        flow_cat = self._categorical_matrix(flow_train, self.flow_cat_cols)

        if packet_cat.shape[1] > 0:
            self.packet_ohe.fit(packet_cat)
        if flow_cat.shape[1] > 0:
            self.flow_ohe.fit(flow_cat)

        self.fitted = True

    def transform_packets(self, packets: pd.DataFrame) -> np.ndarray:
        """
        Transform packet rows into model-ready features.
        """
        self._check_fitted()
        packets = self.add_derived_packet_features(packets)

        parts = []

        packet_num = self._numeric_matrix(packets, self.packet_num_cols)
        if packet_num.shape[1] > 0:
            parts.append(self.packet_scaler.transform(packet_num).astype(np.float32))

        packet_cat = self._categorical_matrix(packets, self.packet_cat_cols)
        if packet_cat.shape[1] > 0:
            parts.append(self.packet_ohe.transform(packet_cat).astype(np.float32))

        return np.concatenate(parts, axis=1).astype(np.float32)

    def transform_flows(self, flows: pd.DataFrame) -> np.ndarray:
        """
        Transform flow rows into model-ready features.
        """
        self._check_fitted()

        parts = []

        flow_num = self._numeric_matrix(flows, self.flow_num_cols)
        if flow_num.shape[1] > 0:
            parts.append(self.flow_scaler.transform(flow_num).astype(np.float32))

        flow_cat = self._categorical_matrix(flows, self.flow_cat_cols)
        if flow_cat.shape[1] > 0:
            parts.append(self.flow_ohe.transform(flow_cat).astype(np.float32))

        return np.concatenate(parts, axis=1).astype(np.float32)

    def input_dim(self) -> int:
        """
        Final x_i,t dimension:
            packet_feature_dim + flow_feature_dim
        """
        return self.packet_feature_dim() + self.flow_feature_dim()

    def packet_feature_dim(self) -> int:
        dim = len(self.packet_num_cols)
        if self.packet_cat_cols:
            dim += len(self.packet_ohe.get_feature_names_out())
        return int(dim)

    def flow_feature_dim(self) -> int:
        dim = len(self.flow_num_cols)
        if self.flow_cat_cols:
            dim += len(self.flow_ohe.get_feature_names_out())
        return int(dim)

    def summary(self) -> Dict[str, Any]:
        return {
            "packet_numerical_used": self.packet_num_cols,
            "packet_categorical_used": self.packet_cat_cols,
            "flow_numerical_used": self.flow_num_cols,
            "flow_categorical_used": self.flow_cat_cols,
            "packet_feature_dim": self.packet_feature_dim(),
            "flow_feature_dim": self.flow_feature_dim(),
            "input_dim": self.input_dim(),
        }

    def _resolve_columns(
        self,
        df: pd.DataFrame,
        requested: List[str],
        group_name: str,
    ) -> List[str]:
        present = [c for c in requested if c in df.columns]
        missing = [c for c in requested if c not in df.columns]

        if missing and self.strict_schema:
            raise ValueError(f"Missing configured {group_name} columns: {missing}")

        return present

    @staticmethod
    def _numeric_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        if not cols:
            return np.zeros((len(df), 0), dtype=np.float32)

        x = df[cols].apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
        return x.astype(np.float32).to_numpy()

    @staticmethod
    def _categorical_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        if not cols:
            return np.zeros((len(df), 0), dtype=object)

        x = df[cols].copy()
        x = x.fillna("MISSING").astype(str)
        return x.to_numpy()

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Stage1Preprocessor is not fitted.")
