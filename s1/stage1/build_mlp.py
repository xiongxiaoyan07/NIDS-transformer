import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()

    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()

    raise ValueError(f"Unsupported activation: {name}")


def _build_norm(norm: str | None, dim: int) -> nn.Module | None:
    if norm is None:
        return None

    norm = norm.lower()

    if norm in ["none", "null", ""]:
        return None
    if norm == "layernorm":
        return nn.LayerNorm(dim)
    if norm == "batchnorm":
        return nn.BatchNorm1d(dim)

    raise ValueError(f"Unsupported norm: {norm}")


def _to_hidden_dims(value):
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)

    raise TypeError(f"hidden_dims should be int/list/tuple/None, got {type(value)}")


def build_mlp(
    input_dim: int,
    output_dim: int,
    mlp_cfg: dict | None,
    default_dropout: float = 0.0,
    legacy_layer_factory=None,
) -> nn.Module:
    """
    Backward-compatible MLP builder.

    - mlp_cfg is None: use legacy_layer_factory if provided, otherwise Linear(input_dim, output_dim)
    - hidden_dims missing: same as legacy mode
    - hidden_dims: []: Linear(input_dim, output_dim)
    - hidden_dims: [h1, h2]: Linear -> Norm -> Act -> Dropout ... -> Linear(output_dim)
    """

    if mlp_cfg is None:
        if legacy_layer_factory is not None:
            return legacy_layer_factory()
        return nn.Linear(input_dim, output_dim)

    hidden_dims = _to_hidden_dims(mlp_cfg.get("hidden_dims", None))

    # 兼容旧配置：没有 hidden_dims 时，不改变旧行为
    if hidden_dims is None:
        if legacy_layer_factory is not None:
            return legacy_layer_factory()
        return nn.Linear(input_dim, output_dim)

    activation = mlp_cfg.get("activation", "gelu")
    dropout = mlp_cfg.get("dropout", default_dropout)
    norm = mlp_cfg.get("norm", None)

    dims = [input_dim] + hidden_dims + [output_dim]

    layers: list[nn.Module] = []

    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]
        is_last = i == len(dims) - 2

        layers.append(nn.Linear(in_dim, out_dim))

        if not is_last:
            norm_layer = _build_norm(norm, out_dim)
            if norm_layer is not None:
                layers.append(norm_layer)

            layers.append(_get_activation(activation))

            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)