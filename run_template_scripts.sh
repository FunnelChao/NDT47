CUDA_VISIBLE_DEVICES=1 python train.py --name "(1208)_(1207)_zscore" --normalize_method zscore --trainval_root_dirs '1208' --test_root_dirs '1207'

CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205)_(1207)_zscore" --normalize_method zscore --trainval_root_dirs '1205' --test_root_dirs '1207'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205)_(1207)_minmax" --normalize_method minmax --trainval_root_dirs '1205' --test_root_dirs '1207'

CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205)_(1208)_zscore" --normalize_method zscore --trainval_root_dirs '1205' --test_root_dirs '1208'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205)_(1208)_minmax" --normalize_method minmax --trainval_root_dirs '1205' --test_root_dirs '1208'

CUDA_VISIBLE_DEVICES=0 python train.py --name "(1205)_(1207,1208)_zscore" --normalize_method zscore --trainval_root_dirs '1205' --test_root_dirs '1207,1208'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205)_(1207,1208)_minmax" --normalize_method minmax --trainval_root_dirs '1205' --test_root_dirs '1207,1208'

CUDA_VISIBLE_DEVICES=0 python train.py --name "(1208)_(1205)_zscore" --normalize_method zscore --trainval_root_dirs '1208' --test_root_dirs '1205'

CUDA_VISIBLE_DEVICES=1 python train.py --name "(1208)_(1205,1207)_zscore" --normalize_method zscore --trainval_root_dirs '1208' --test_root_dirs '1207,1205'

CUDA_VISIBLE_DEVICES=0 python train.py --name "(1205,1208)_(1207)_zscore" --normalize_method zscore --trainval_root_dirs '1205,1208' --test_root_dirs '1207'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205,1208)_(1207)_minmax" --normalize_method minmax --trainval_root_dirs '1205,1208' --test_root_dirs '1207'

CUDA_VISIBLE_DEVICES=0 python train.py --name "(1207,1208)_(1205)_zscore" --normalize_method zscore --trainval_root_dirs '1207,1208' --test_root_dirs '1205'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1207,1208)_(1205)_minmax" --normalize_method minmax --trainval_root_dirs '1207,1208' --test_root_dirs '1205'



CUDA_VISIBLE_DEVICES=1 python train.py --name "(1205)_(1207)_zscore_1e-5" --normalize_method zscore --trainval_root_dirs '1205' --test_root_dirs '1207'

CUDA_VISIBLE_DEVICES=1 python train.py --name "(1207)_(1205)_zscore" --normalize_method zscore --trainval_root_dirs '1207' --test_root_dirs '1205'
CUDA_VISIBLE_DEVICES=1 python train.py --name "(1207)_(1205)_minmax" --normalize_method minmax --trainval_root_dirs '1207' --test_root_dirs '1205'


CUDA_VISIBLE_DEVICES=1 python train.py --name "(1207)_(1208)_zscore" --normalize_method zscore --trainval_root_dirs '1207' --test_root_dirs '1208'
CUDA_VISIBLE_DEVICES=0 python train.py --name "(1207)_(1208)_minmax" --normalize_method minmax --trainval_root_dirs '1207' --test_root_dirs '1208'