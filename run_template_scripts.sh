# qianqian
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205,1207)_(1208)_zscore_0" --normalize_method zscore --cfg config/cross_day_0.yaml

CUDA_VISIBLE_DEVICES=0 python train.py --name "(1205)_(1207)_zscore_0" --normalize_method zscore --cfg 'config/cross_day_(1205)_(1207)_0.yaml'

CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205_0)_(1208_60,-60,120,-120)_zscore" --normalize_method zscore --cfg 'config/qianqian_cross_day_(1205_0)_(1208_60,-60,120,-120).yaml'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205_0)_(1205_60,-60,120,-120)_zscore" --normalize_method zscore --cfg 'config/qianqian_cross_angle_(1205_0)_(1205_60,-60,120,-120).yaml'

CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205_0)_(1208)_zscore" --normalize_method zscore --cfg 'config/qianqian_cross_day_(1205_0)_(1208).yaml'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205,1207)_(1208)_zscore" --normalize_method zscore --cfg 'config/qianqian_cross_day_(1205,1207)_(1208).yaml'

CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205,1208)_(1207)_12layer_zscore" --normalize_method zscore --cfg 'config/qianqian_cross_day_(1205,1208)_(1207).yaml' --enc_layers 12
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205,1207)_(1208)_12layer_zscore" --normalize_method zscore --cfg 'config/qianqian_cross_day_(1205,1207)_(1208).yaml' --enc_layers 12
CUDA_VISIBLE_DEVICES=0 python train.py --name "(1207,1208)_(1205)_12layer_zscore" --normalize_method zscore --cfg 'config/qianqian_cross_day_(1207,1208)_(1205).yaml' --enc_layers 12


# nezha
CUDA_VISIBLE_DEVICES=0 python train.py --name "(0314)_(0315,0320,0321,0322)_zscore" --normalize_method zscore --cfg 'config/nezha_cross_day_(0314)_(0315,0320,0321,0322).yaml'

CUDA_VISIBLE_DEVICES=0 python train.py --name "(0314,0320,0321,0322)_(0315)_zscore" --normalize_method zscore --cfg 'config/nezha_cross_day_(0314,0320,0321,0322)_(0315).yaml'

CUDA_VISIBLE_DEVICES=0 python train.py --name "(0315,0320,0321,0322)_(0314)_zscore" --normalize_method zscore --cfg 'config/nezha_cross_day_(0315,0320,0321,0322)_(0314).yaml'
CUDA_VISIBLE_DEVICES=0 python train.py --name "(0314,0315,0321,0322)_(0320)_zscore" --normalize_method zscore --cfg 'config/nezha_cross_day_(0314,0315,0321,0322)_(0320).yaml'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(0314,0315,0320,0322)_(0321)_zscore" --normalize_method zscore --cfg 'config/nezha_cross_day_(0314,0315,0320,0322)_(0321).yaml'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(0314,0315,0320,0321)_(0322)_zscore" --normalize_method zscore --cfg 'config/nezha_cross_day_(0314,0315,0320,0321)_(0322).yaml'

# Bohr
CUDA_VISIBLE_DEVICES=0 python train.py --name "Bohr(0402_0)_(0402_90,-90,180,-180)_zscore" --normalize_method zscore --cfg 'config/Bohr_cross_angle_(0402_0)_(0402_90,-90,180,-180).yaml'
CUDA_VISIBLE_DEVICES=0 python train.py --name "Bohr(0402_0,-180,90)_(0402_-90,180)_zscore" --normalize_method zscore --cfg 'config/Bohr_cross_angle_(0402_0,-180,90)_(0402_-90,180).yaml'

CUDA_VISIBLE_DEVICES=0 python train.py --name "Bohr(0402)_zscore" --normalize_method zscore --cfg 'config/Bohr_(0402).yaml'