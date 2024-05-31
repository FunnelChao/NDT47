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
from dataset import *
from model import NDT47
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
    criterion.train()

    log_freq = 10

    train_targets = []
    train_preds = []
    running_loss = []
    pbar = tqdm(data_loader,colour='red')
    for idx, samples in enumerate(pbar):
        samples = {k: v.to(device) for k, v in samples.items()}
        targets = samples['pos']
        outputs = model(samples)
        losses= criterion(outputs, targets)
        loss_value = losses.item()
        running_loss.append(loss_value)
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        train_targets.append(targets.cpu())
        train_preds.append(outputs.detach().cpu())

        r2 = r_squared(y_true=torch.tensor(data_loader.dataset.scaler.inverse_transform(torch.concat(train_targets))), 
                       y_pred=torch.tensor(data_loader.dataset.scaler.inverse_transform(torch.concat(train_preds)))
                       )
        pbar.set_description(f'train:[Epoch:{epoch}]: [loss:{np.mean(running_loss):.5f}] [R2:{r2:.4f}]')
        if log_freq % log_freq:
            pass

        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    epoch_metric = {
        f'train_loss':np.mean(running_loss),
        f'train_R2':r2
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
    criterion.eval()

    val_targets = []
    val_preds = []
    running_loss = []
    phase = display_config['phase']
    log_color = display_config['color']
    pbar = tqdm(data_loader, colour=log_color)
    for idx, samples in enumerate(pbar):
        samples = {k: v.to(device) for k, v in samples.items()}
        targets = samples['pos']
        outputs = model(samples)
        losses= criterion(outputs, targets)
        loss_value = losses.item()
        running_loss.append(loss_value)

        val_targets.append(targets.cpu())
        val_preds.append(outputs.detach().cpu())

        r2 = r_squared(y_true=torch.tensor(data_loader.dataset.scaler.inverse_transform(torch.concat(val_targets))), 
                       y_pred=torch.tensor(data_loader.dataset.scaler.inverse_transform(torch.concat(val_preds)))
                       )
        pbar.set_description(f'{phase.center(5)}:[Epoch:{epoch}]: [loss:{np.mean(running_loss):.5f}] [R2:{r2:.4f}]')

    epoch_metric = {
        f'{phase}_loss':np.mean(running_loss),
        f'{phase}_R2':r2
    }
    wandb.log(epoch_metric, step=epoch)



def get_args_parser():
    parser = argparse.ArgumentParser('Set NDT47', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=500, type=int)
    parser.add_argument('--clip_max_norm', default=1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
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
    parser.add_argument('--cfg', default='config/qianqian_cross_day_(1205)_(1207)_0.yaml')
    parser.add_argument('--normalize_method', default='minmax',
                        help='')

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
    train_data, val_data = load_data(cfg=data_cfg['train'], type=data_cfg['type'], phase='trainval', train_val_rate=0.7, seed=1)
    test_data = load_data(cfg=data_cfg['test'], type=data_cfg['type'], phase='test', seed=1)

    train_dataset = NDTDataset(data=train_data, normalize_method=args.normalize_method)
    val_dataset = NDTDataset(data=val_data, normalize_method=args.normalize_method, scaler=train_dataset.scaler)
    test_dataset = NDTDataset(data=test_data, normalize_method=args.normalize_method, scaler=train_dataset.scaler)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)  
    # ----------------------------------------------------------------------model config--------------------------------------------------------------------
    model = NDT47(input_dim=data_cfg['channel'], 
                  d_model=args.hidden_dim, 
                  nhead=args.nheads, 
                  num_encoder_layers=args.enc_layers,
                  dim_feedforward=args.dim_feedforward, 
                  dropout=args.dropout, 
                  d_out=args.out_dim,
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
    # -----------------------------------------------------------------------parameters config---------------------------------------------------------------
    criterion = nn.MSELoss() 

    # -----------------------------------------------------------------------training---------------------------------------------------------------
    print("Start training")
    import wandb
    wandb.login()
    wandb.init(project="NDT47", name=args.name)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args.clip_max_norm)
        evaluate(model, criterion, val_loader, device, epoch, display_config={'color':'blue', 'phase':'val'})
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