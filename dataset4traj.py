import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import seed_everything
import numpy as np
import random
from utils import r_squared, load_cfg
#---------------------------------------------------------------------------------------------for decoding trajectory---------------------------------------------------------------------------------------------
def load_data4traj(cfg, type='unsort', phase='trainval', train_val_rate=0.7, seed=1):
    """
    return : N * [path]
    """
    seed_everything(seed)
    total = []

    data_statistics = {}
    for obj in cfg.keys():
        data_statistics[obj] = {}
        for date in cfg[obj]:
            data_statistics[obj][date] = {}
            valid_trial_index = [i.split('_')[-1].split('.')[0] for i in os.listdir(os.path.join(f'data/cui/{obj}/{date}', "hand_traj"))]
            if cfg[obj][date]['v_angle'] == 'all':
                v_angle_list = os.listdir(f'data/cui/{obj}/{date}/psth_{type}')
            else:
                v_angle_list = cfg[obj][date]['v_angle']
            for v in v_angle_list:
                samples = glob.glob(os.path.join(f'data/cui/{obj}/{date}/psth_{type}/{v}', "psth_trail_*"))
                samples_trials_index = [i.split('_')[-1].split('.')[0] for i in samples]
                valid_trial_index = [int(i) for i in samples_trials_index if i in valid_trial_index]
                valid_samples = [f'data/cui/{obj}/{date}/psth_{type}/{v}/psth_trail_{i}.csv' for i in valid_trial_index]
                valid_samples_traj = [f'data/cui/{obj}/{date}/hand_traj/hand_traj_trail_{i}.csv' for i in valid_trial_index]
                data_statistics[obj][date][v] = len(valid_samples)
                data = np.concatenate([np.array(valid_samples)[:,np.newaxis],np.array(valid_samples_traj)[:,np.newaxis]],axis=-1)
                np.random.shuffle(data)
                total.append(data)
        
    assert phase in ['trainval','val','test'], f'phase {phase} is not available'
    if phase == 'trainval':
        train_samples = np.vstack([i[:int(i.shape[0]*train_val_rate)] for i in total])
        val_samples = np.vstack([i[int(i.shape[0]*train_val_rate):] for i in total])
        return train_samples, val_samples
    elif phase == 'val' or phase == 'test':
        return np.vstack(total)

class NDTDataset4traj(Dataset):
    def __init__(self, data, normalize_method='zscore', scaler=None):
        self.samples = data
        if scaler is None:
            all_x_y = [pd.read_csv(i,index_col=0)/10 for i in self.samples[:,1]]
            max_len = max([i.shape[0] for i in all_x_y])
            all_x_y = np.concatenate([pd.read_csv(i,index_col=0)/10 for i in self.samples[:,1]])
            _, scaler = normalize(data=all_x_y, method=normalize_method)
            self.scaler = scaler
            self.max_len = max_len
        else:
            self.scaler = scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_path, traj_path = self.samples[idx]
        data = pd.read_csv(data_path,index_col=0).values.transpose(1,0)
        traj = torch.tensor(self.scaler.transform(pd.read_csv(traj_path,index_col=0).values / 10), dtype=torch.float32)
        return data, traj
    
def collate(data_list):
    bsz = len(data_list)
    lengths = torch.tensor([i[-1].shape[0] for i in data_list], dtype=torch.int)
    
    pos_index = torch.arange(max(lengths)).repeat(bsz,1)
    masks = pos_index > lengths[:,None]
    feature = torch.stack([torch.tensor(i[0], dtype=torch.float32) for i in data_list])
    padded_traj = torch.zeros(bsz, max(lengths), 2)
    for idx, (data, lenth) in enumerate(zip(data_list, lengths)):
        padded_traj[idx][:lenth] = torch.FloatTensor(data[-1])
    to_return = {
        'feat': feature,
        'mask': masks,
        'traj': padded_traj,
        'length': lengths,
        'pos_index': pos_index
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


# debug-------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -----------------------------------------------------------------------data config---------------------------------------------------------------
    data_cfg = load_cfg('config/Bohr_(0402).yaml')
    train_data, val_data = load_data4traj(cfg=data_cfg['train'], type=data_cfg['type'], phase='trainval', train_val_rate=0.7, seed=1)
    a = NDTDataset4traj(train_data, normalize_method='zscore')
    a.__getitem__(0)
    batch_data = collate([a.__getitem__(i) for i in range(8)])