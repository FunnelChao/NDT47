# NDT47
version 1: 默认所有的速度进行训练，同一object跨天训练

## model
<img src="model.png" alt="替代文本" width="300">

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
# train on 0314 data and test on (0315,0320,0321,0322) data directly
CUDA_VISIBLE_DEVICES=0 python train.py --name "(0314)_(0315,0320,0321,0322)_zscore" --normalize_method zscore --cfg 'config/nezha_cross_day_(0314)_(0315,0320,0321,0322).yaml'
```