# Stage-I Suricata Output Analysis Report

## 1. Dataset size

- packets rows: `1,003,219`
- flows rows: `38,671`
- packet columns: `29`
- flow columns: `73`
- unique flow_id in packets: `38,671`
- unique flow_id in flows: `38,671`

## 2. Label distribution

### Packet labels

|   packet_label |   count |
|---------------:|--------:|
|              0 |  991346 |
|              1 |   11873 |

### Flow labels

|   label |   count |
|--------:|--------:|
|       0 |   37380 |
|       1 |    1291 |

- flow attack ratio: `0.033384`

## 3. Protocol distribution

### Flow protocol counts

|   protocol |   count |
|-----------:|--------:|
|          1 |    1126 |
|          6 |   26954 |
|         17 |   10259 |
|         41 |       1 |
|         50 |     325 |
|         58 |       6 |

### Packet protocol counts

|   protocol |   count |
|-----------:|--------:|
|          1 |    6992 |
|          6 |  593077 |
|         17 |  329649 |
|         41 |      60 |
|         50 |   73379 |
|         58 |      62 |

## 4. IAT summary


### `flow_iat_us`

|       |      flow_iat_us |
|:------|-----------------:|
| count |      1.00322e+06 |
| mean  |  54935.4         |
| std   | 375419           |
| min   |      0           |
| 50%   |    202           |
| 90%   |  24023.2         |
| 95%   |  88566.9         |
| 99%   |      1.54366e+06 |
| max   |      6.0912e+06  |

### `direction_iat_us`

|       |   direction_iat_us |
|:------|-------------------:|
| count |        1.00322e+06 |
| mean  |    77614.2         |
| std   |   429950           |
| min   |        0           |
| 50%   |      216           |
| 90%   |    52193.8         |
| 95%   |   231924           |
| 99%   |        2.18466e+06 |
| max   |        6.0912e+06  |

## 5. Flow-packet consistency

- consistency rows: `38,671`

### merge status

| _merge     |   count |
|:-----------|--------:|
| both       |   38671 |
| left_only  |       0 |
| right_only |       0 |
- fwd packet count mismatches: `0`
- bwd packet count mismatches: `0`
- label mismatches: `0`

## 6. Top correlations with flow label

| feature                 |   pearson_corr_with_label |
|:------------------------|--------------------------:|
| init_win_bytes_forward  |                  0.307213 |
| init_win_bytes_backward |                  0.261605 |
| bwd_packet_length_std   |                  0.254234 |
| bwd_packet_length_max   |                  0.253647 |
| fwd_iat_std             |                  0.246209 |
| max_packet_length       |                  0.206449 |
| bwd_iat_max             |                  0.169603 |
| packet_length_std       |                  0.168511 |
| bwd_packet_length_mean  |                  0.167321 |
| avg_bwd_segment_size    |                  0.167321 |
| bwd_iat_std             |                  0.166484 |
| flow_iat_std            |                  0.164482 |
| fwd_iat_max             |                  0.147327 |
| fwd_iat_total           |                  0.147231 |
| flow_iat_max            |                  0.14389  |
| flow_duration           |                  0.143829 |
| packet_length_variance  |                  0.139485 |
| bwd_iat_total           |                  0.138313 |
| source_port             |                  0.130364 |
| active_max              |                  0.129416 |
| active_mean             |                  0.128811 |
| active_min              |                  0.128203 |
| fwd_packet_length_max   |                  0.119537 |
| fin_flag_count          |                  0.114532 |
| syn_flag_count          |                  0.109587 |

## 7. Generated files

- `numeric_summary_packets.csv`
- `numeric_summary_flows.csv`
- `missing_values_packets.csv`
- `missing_values_flows.csv`
- `flow_packet_consistency.csv`
- `label_consistency.csv`
- `top_label_correlations.csv`
- `stage1_packets_enriched.csv`
- `stage1_flows_ml_features.csv`
- `plots/*.png`