---
dataset_info:
  features:
  - name: target
    dtype: int64
  - name: nums
    sequence: int64
  splits:
  - name: train
    num_bytes: 19650960
    num_examples: 490364
  download_size: 2845904
  dataset_size: 19650960
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
