CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205,1207)_(1208)_zscore_0" --normalize_method zscore --cfg config/cross_day_0.yaml

CUDA_VISIBLE_DEVICES=0 python train.py --name "(1205)_(1207)_zscore_0" --normalize_method zscore --cfg 'config/cross_day_(1205)_(1207)_0.yaml'

CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205_0)_(1208_60,-60,120,-120)_zscore" --normalize_method zscore --cfg 'config/qianqian_cross_day_(1205_0)_(1208_60,-60,120,-120).yaml'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205_0)_(1205_60,-60,120,-120)_zscore" --normalize_method zscore --cfg 'config/qianqian_cross_angle_(1205_0)_(1205_60,-60,120,-120).yaml'

CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205_0)_(1208)_zscore" --normalize_method zscore --cfg 'config/qianqian_cross_day_(1205_0)_(1208).yaml'


CUDA_VISIBLE_DEVICES=0 python train.py --name "(0314)_(0315,0320,0321,0322)_zscore" --normalize_method zscore --cfg 'config/nezha_cross_day_(0314)_(0315,0320,0321,0322).yaml'

CUDA_VISIBLE_DEVICES=0 python train.py --name "(0314,0320,0321,0322)_(0315)_zscore" --normalize_method zscore --cfg 'config/nezha_cross_day_(0314,0320,0321,0322)_(0315).yaml'