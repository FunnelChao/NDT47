import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import seed_everything
import numpy as np



def load_data(root_dirs, phase='trainval', train_val_rate=0.7, seed=1):
    seed_everything(seed)
    total = []
    root_dirs = root_dirs.split(',')
    for dir in root_dirs:
        samples = glob.glob(os.path.join(f'data/cui/{dir}_psth_TCR', "psth_trail_*"))
        samples.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
        pos = pd.read_csv(os.path.join(f'data/cui/{dir}_psth_TCR', "end_pos.csv"),index_col=0).sort_values('TrialIndex').values[:,:2]
        sample_pos = np.concatenate([np.array(samples)[:,np.newaxis],pos],axis=-1)
        np.random.shuffle(sample_pos)
        total.append(sample_pos)
        
    assert phase in ['trainval','val','test'], f'phase {phase} is not available'
    if phase == 'trainval':
        train_samples = np.vstack([i[:int(i.shape[0]*train_val_rate)] for i in total])
        val_samples = np.vstack([i[int(i.shape[0]*train_val_rate):] for i in total])
        return train_samples, val_samples
    elif phase == 'val' or phase == 'test':
        return np.vstack(total)


class NDTDataset(Dataset):
    def __init__(self, data, normalize_method='zscore', scaler=None):
        self.samples = data
        if scaler is None:
            normed_xy, scaler = normalize(data=data[:,1:], method=normalize_method)
            self.samples[:,1:] = normed_xy
            self.scaler = scaler
        else:
            self.scaler = scaler
            self.samples[:,1:] = scaler.fit_transform(data[:,1:])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, x, y = self.samples[idx]
        sample_data = pd.read_csv(sample_path,index_col=0).values.transpose(1,0)
        target_pos = torch.tensor([np.float32(x),np.float32(y)], dtype=torch.float32)

        return sample_data, target_pos
    
def collate(data_list):
    bsz = len(data_list)
    feat_dim = data_list[0][0].shape[1]
    lengths = [i[0].shape[0] for i in data_list]
    targets = torch.stack([i[1] for i in data_list])
    masks = torch.stack([torch.arange(max(lengths)) for i in range(bsz)])
    masks = masks > (torch.tensor(lengths)[:,None])
    aligned_feature = torch.zeros(bsz, max(lengths), feat_dim)
    for idx, (data, lenth) in enumerate(zip(data_list, lengths)):
        aligned_feature[idx][:lenth] = torch.FloatTensor(data_list[idx][0])
    to_return = {
        'feat':aligned_feature,
        'mask':masks,
        'pos':targets
    }
    return to_return

def normalize(data, method='zscore'):
    if method=='zscore':
        from sklearn.preprocessing import StandardScaler
        # 创建 StandardScaler 对象
        scaler = StandardScaler()
        # 训练并应用 Z-Score 归一化
        scaled_data = scaler.fit_transform(data)
        # # 反归一化已归一化的数据
        # original_data = scaler.inverse_transform(scaled_data)
    elif method=='minmax':
        from sklearn.preprocessing import MinMaxScaler
        # 创建 MinMaxScaler 对象
        scaler = MinMaxScaler()
        # 训练并应用归一化到指定范围
        scaled_data = scaler.fit_transform(data)
        # # 反归一化已归一化的数据
        # original_data = scaler.inverse_transform(scaled_data)
    return scaled_data, scaler




if __name__ == '__main__':

    # debug 
    root_dirs = '1205,1207'

    train_data, val_data = load_data(root_dirs=root_dirs, phase='trainval', train_val_rate=0.7, seed=1)
    a = NDTDataset(train_data, normalize_method='zscore')
    a.__getitem__(0)
    batch_data = collate([a.__getitem__(i) for i in range(8)])