import argparse
import datetime
import json
import random
import time
from pathlib import Path
from typing import Iterable
import numpy as np
import math
import torch
from torch.utils.data import DataLoader, DistributedSampler
from dataset4traj import *
from model import NDT47_for_traj
import torch.nn as nn
from tqdm import tqdm
from utils import r_squared, load_cfg
import wandb

def train_one_epoch(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int,
                    max_norm: float = 0,
                    ):
    model.train()
    criterion['reg'].train()
    criterion['cls'].train()
    log_freq = 10

    # record regression status
    running_traj_targets = []
    running_traj_preds = []
    running_regre_loss = []

    # record classification status
    running_cls_loss = []
    running_cls_hit = 0
    running_sample_num = 0

    # record total loss
    running_total_loss = []
    pbar = tqdm(data_loader,colour='red')
    for idx, samples in enumerate(pbar):
        samples = {k: v.to(device) for k, v in samples.items()}
        tgt_end_index = samples['length'] - 1
        decode_traj, is_end = model(samples)

        # classification
        pred_end_logits = nn.Softmax(dim=-1)(is_end.squeeze(-1))
        pred_end_index = torch.argmax(pred_end_logits,dim=-1)
        cls_loss = criterion['cls'](pred_end_logits, tgt_end_index.long())
        running_cls_hit += torch.sum(pred_end_index==tgt_end_index)
        running_sample_num += samples['length'].shape[0]


        # regression
        pred_traj = torch.concat([i[:len] for i,len in zip(decode_traj, samples['length'])])
        gt_traj = torch.concat([i[:len] for i,len in zip(samples['traj'], samples['length'])])
        reg_loss = criterion['reg'](pred_traj, gt_traj)

        total_loss = criterion['loss_weight']['reg']*reg_loss + criterion['loss_weight']['cls']*cls_loss

        running_total_loss.append(total_loss.item())
        running_regre_loss.append(reg_loss.item())
        running_cls_loss.append(cls_loss.item())
        
        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        running_traj_targets.append(gt_traj.cpu())
        running_traj_preds.append(pred_traj.detach().cpu())

        r2 = r_squared(y_true=torch.tensor(data_loader.dataset.scaler.inverse_transform(torch.concat(running_traj_targets))), 
                       y_pred=torch.tensor(data_loader.dataset.scaler.inverse_transform(torch.concat(running_traj_preds)))
                       )
        pbar.set_description(f'train:[Epoch:{epoch}]:[toatl_loss:{np.mean(running_total_loss):.5f}][regre_loss:{np.mean(running_regre_loss):.5f}][cls_loss:{np.mean(running_cls_loss):.5f}][R2:{r2:.4f}][cls_acc:{running_cls_hit/running_sample_num:.4f}]')
        if log_freq % log_freq:
            pass

    epoch_metric = {
        'epoch':epoch,
        'toatl_loss':np.mean(running_total_loss),
        'regre_loss':np.mean(running_regre_loss),
        'cls_loss':np.mean(running_cls_loss),
        'R2':r2,
        'cls_acc':running_cls_hit/running_sample_num
    }
    wandb.log(epoch_metric, step=epoch)

@torch.no_grad()
def evaluate(
    model: torch.nn.Module, 
    criterion: torch.nn.Module,
    data_loader: Iterable, 
    device: torch.device, 
    epoch: int,
    display_config: list
    ):
    model.eval()
    criterion['reg'].eval()
    criterion['cls'].eval()

    # record regression status
    running_traj_targets = []
    running_traj_preds = []
    running_regre_loss = []

    # record classification status
    running_cls_loss = []
    running_cls_hit = 0
    running_sample_num = 0

    # record total loss
    running_total_loss = []

    phase = display_config['phase']
    log_color = display_config['color']
    pbar = tqdm(data_loader, colour=log_color)
    for idx, samples in enumerate(pbar):
        samples = {k: v.to(device) for k, v in samples.items()}
        tgt_end_index = samples['length'] - 1
        decode_traj, is_end = model(samples)

        # classification
        pred_end_logits = nn.Softmax(dim=-1)(is_end.squeeze(-1))
        pred_end_index = torch.argmax(pred_end_logits,dim=-1)
        cls_loss = criterion['cls'](pred_end_logits, tgt_end_index.long())
        running_cls_hit += torch.sum(pred_end_index==tgt_end_index)
        running_sample_num += samples['length'].shape[0]


        # regression
        pred_traj = torch.concat([i[:len] for i,len in zip(decode_traj, samples['length'])])
        gt_traj = torch.concat([i[:len] for i,len in zip(samples['traj'], samples['length'])])
        reg_loss = criterion['reg'](pred_traj, gt_traj)

        total_loss = criterion['loss_weight']['reg']*reg_loss + criterion['loss_weight']['cls']*cls_loss

        running_total_loss.append(total_loss.item())
        running_regre_loss.append(reg_loss.item())
        running_cls_loss.append(cls_loss.item())
        
        running_traj_targets.append(gt_traj.cpu())
        running_traj_preds.append(pred_traj.detach().cpu())

        r2 = r_squared(y_true=torch.tensor(data_loader.dataset.scaler.inverse_transform(torch.concat(running_traj_targets))), 
                       y_pred=torch.tensor(data_loader.dataset.scaler.inverse_transform(torch.concat(running_traj_preds)))
                       )
        pbar.set_description(f'{phase.center(5)}:[Epoch:{epoch}]:[toatl_loss:{np.mean(running_total_loss):.5f}][regre_loss:{np.mean(running_regre_loss):.5f}][cls_loss:{np.mean(running_cls_loss):.5f}][R2:{r2:.4f}][cls_acc:{running_cls_hit/running_sample_num:.4f}]')

    epoch_metric = {
        f'epoch':epoch,
        f'{phase}_toatl_loss':np.mean(running_total_loss),
        f'{phase}_regre_loss':np.mean(running_regre_loss),
        f'{phase}_cls_loss':np.mean(running_cls_loss),
        f'{phase}_R2':r2,
        f'{phase}_cls_acc':running_cls_hit/running_sample_num
    }
    wandb.log(epoch_metric, step=epoch)




def get_args_parser():
    # optimizer 
    parser = argparse.ArgumentParser('Set NDT47_for_traj', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_drop', default=1000, type=int)
    parser.add_argument('--clip_max_norm', default=1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--reg_loss_w', default=1, type=float)
    parser.add_argument('--cls_loss_w', default=1, type=float)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--max_query_size', default=100, type=int,
                        help="Size of the output")
    parser.add_argument('--input_dim', default=256, type=int,
                        help="Size of the input")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--out_dim', default=2, type=int,
                        help="Size of the output")
    parser.add_argument('--pre_norm', action='store_true')

    # dataset parameters
    parser.add_argument('--cfg', default='config/Bohr_(0402).yaml')
    parser.add_argument('--normalize_method', default='zscore',
                        help='zscore/minmax')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)

    # log setting
    parser.add_argument('--name', default=None, type=str,
                    help='wandb running name')
    return parser


def main(args):
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # -----------------------------------------------------------------------data config---------------------------------------------------------------
    data_cfg = load_cfg(args.cfg)
    train_data, val_data = load_data4traj(cfg=data_cfg['train'], type=data_cfg['type'], phase='trainval', train_val_rate=0.7, seed=1)
    

    train_dataset = NDTDataset4traj(data=train_data, normalize_method=args.normalize_method)
    val_dataset = NDTDataset4traj(data=val_data, normalize_method=args.normalize_method, scaler=train_dataset.scaler)
    

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    
    if data_cfg.get('test'):
        test_data = load_data4traj(cfg=data_cfg['test'], type=data_cfg['type'], phase='test', seed=1)
        test_dataset = NDTDataset4traj(data=test_data, normalize_method=args.normalize_method, scaler=train_dataset.scaler)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)  
    # ----------------------------------------------------------------------model config--------------------------------------------------------------------
    model = NDT47_for_traj(input_dim=data_cfg['channel'], 
                  d_model=args.hidden_dim, 
                  nhead=args.nheads, 
                  num_encoder_layers=args.enc_layers,
                  num_decoder_layers=args.dec_layers,
                  dim_feedforward=args.dim_feedforward, 
                  dropout=args.dropout, 
                  d_out=args.out_dim,
                  max_query_size=args.max_query_size,
                  activation="relu", 
                  normalize_before=False)
    model.to(device)

    # ----------------------------------------------------------------------pretrain config--------------------------------------------------------------------
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        return

    # -----------------------------------------------------------------------parameters config---------------------------------------------------------------
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        # {
        #     "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        #     "lr": args.lr_backbone,
        # },
    ]
    # -----------------------------------------------------------------------optimizer config---------------------------------------------------------------
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # -----------------------------------------------------------------------loss config---------------------------------------------------------------
    criterion_reg = nn.MSELoss() 
    criterion_cls = nn.CrossEntropyLoss() 
    criterion = {
        'reg':criterion_reg,
        'cls':criterion_cls,
        'loss_weight':{
            'reg':args.reg_loss_w,
            'cls':args.cls_loss_w
        }
    }
    # -----------------------------------------------------------------------training---------------------------------------------------------------
    print("Start training")
    import wandb
    wandb.login()
    wandb.init(project="NDT47_for_traj", name=args.name)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args.clip_max_norm)
        evaluate(model, criterion, val_loader, device, epoch, display_config={'color':'blue', 'phase':'val'})
        if data_cfg.get('test'):
            evaluate(model, criterion, test_loader, device, epoch, display_config={'color':'yellow', 'phase':'test'})
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('NDT47 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)