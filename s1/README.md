# Stage1 Modular Pipeline

This project implements Stage1 for a packet-sequence based NIDS pipeline.

It uses:
- `stage1_packets.csv` as packet-level sequence input.
- `stage1_flows.csv` as flow-level statistical context and labels.
- configurable packet and flow feature lists in `configs/stage1_config.yaml`.
- stratified train/val/test split, or an external final test set.
- one-hot encoding for categorical features.
- standard normalization for numerical features.
- record-level projection.
- positional + `flow_iat_us` time-aware encoding.
- Transformer encoder for intra-flow packet modeling.

## Basic training

```bash
python run_stage1.py \
  --packet_csv /home/xxiong/pcaps/stage1_packets.csv \
  --flow_csv /home/xxiong/pcaps/stage1_flows.csv \
  --config configs/stage1_config.yaml \
  --out_dir ./stage1_artifacts
```

## Use another file as final test set

The external test set must have the same schema or at least the configured columns.

```bash
python run_stage1.py \
  --packet_csv ./train_stage1_packets.csv \
  --flow_csv ./train_stage1_flows.csv \
  --external_packet_csv ./final_test_packets.csv \
  --external_flow_csv ./final_test_flows.csv \
  --config configs/stage1_config.yaml \
  --out_dir ./stage1_artifacts_external_test
```

When external test files are provided:
- scalers and one-hot encoders are fitted only on the train split of the training CSVs.
- external test data is transformed using the training preprocessor.
- no external test statistics are used during fitting.

## Most important configurable fields

Edit `configs/stage1_config.yaml`:

```yaml
features:
  flow:
    numerical:
      - flow_duration
      - total_fwd_packets
      - packet_length_mean
      - ...
    categorical:
      - protocol
      - has_init_win_bytes_forward
      - has_init_win_bytes_backward
```

You can remove or add flow fields without changing Python code.
Missing fields are skipped by default if `strict_schema: false`.
