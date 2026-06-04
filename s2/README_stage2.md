# Stage2 Inter-flow Context Transformer

This package trains Stage2 from frozen Stage1 flow embeddings.

Stage1 already exports `z_i` via `export_embeddings.py` as `.npz` files with:

- `flow_id`
- `label`
- `z`

Stage2 loads these embeddings, attaches flow metadata from the original `flow_csv`, constructs an inter-flow context sequence, and trains a second Transformer classifier.

## Run Stage1 with embedding export

```bash
python run_stage1.py \
  --packet_csv ./stage1_packets.csv \
  --flow_csv ./stage1_flows.csv \
  --config ./stage1_full_64_head_both_C_config.yaml \
  --out_dir ./stage1_artifacts \
  --export_embeddings
```

This should produce:

```text
stage1_artifacts/stage1_train_embeddings.npz
stage1_artifacts/stage1_val_embeddings.npz
stage1_artifacts/stage1_test_embeddings.npz
```

## Run Stage2

```bash
cd stage2_pkg
python run_stage2.py \
  --config ./stage2_config.yaml \
  --out_dir ./stage2_artifacts_time_only
```

## Ablation configs

### 1. No context baseline

```yaml
context:
  mode: no_context
  max_context_len: 1
```

### 2. Time-only window

```yaml
context:
  mode: time_only
  direction: past
  max_context_len: 16
  time_window: null
```

### 3. Host-aware source context

```yaml
context:
  mode: host_aware
  host_relation: source
  direction: past
  max_context_len: 16
```

### 4. Host-aware destination context

```yaml
context:
  mode: host_aware
  host_relation: destination
  direction: past
  max_context_len: 16
```

### 5. Host-aware endpoint context

```yaml
context:
  mode: host_aware
  host_relation: endpoint
  direction: past
  max_context_len: 16
```

## Context definitions

Let the target flow be `F_i = (src_i, dst_i, t_i)`.

- Time-only past context:

```text
C_i = {F_j | t_j <= t_i}
```

The nearest `max_context_len` flows are selected by temporal distance and then ordered by time.

- Source-host context:

```text
C_i = {F_j | src_j = src_i and t_j <= t_i}
```

- Destination-host context:

```text
C_i = {F_j | dst_j = dst_i and t_j <= t_i}
```

- Endpoint-aware context:

```text
C_i = {F_j | src_j or dst_j shares src_i or dst_i, and t_j <= t_i}
```

`direction: symmetric` is supported for offline experiments but can use future flows, so `direction: past` is recommended for realistic NIDS evaluation.
