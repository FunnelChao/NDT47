# NDT47
version 1: 默认所有的速度进行训练，同一object跨天训练

## data preparation
```
unzip data.zip
```
and data folder is like this:
```
data/
└── 1205_psth_TCR/
    └── end_pos.csv
    └── psth_trial_1.csv
        ...
    └── psth_trial_n.csv
└── 1207_psth_TCR/
    ...
└── 1208_psth_TCR/
    ...
        	
```

## train
```python
# train on 1208 data and test on 1207 data directly
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1208)_(1205)_zscore" --normalize_method zscore --trainval_root_dirs '1208' --test_root_dirs '1205'

```