import warnings
warnings.filterwarnings('ignore')

import argparse
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from init_srcnn import TrainDataset, EvalDataset, calc_psnr, AverageMeter
from SRCNN import SRCNN

# eval:
# 2x: 33.26
# 4x: 27.26

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='/media/gallade/disk2/AID/train')
    parser.add_argument('--valid_dir', type=str, default='/media/gallade/disk2/AID/valid')
    parser.add_argument('--save_weight_dir', type=str, default='SRCNNx4.pth')  ###
    parser.add_argument('--upscale_factor', type=int, default=4)  ###
    parser.add_argument('--preupsampling', type=bool, default=True)  ###
    parser.add_argument('--lr', type=float, default=5e-4)  # SRCNN:5e-4  RCAN:1e-4
    parser.add_argument('--train_patch_size', type=int, default=32) # x2:32  x4:64  x8:128
    parser.add_argument('--valid_patch_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--thread', type=int, default=32)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.thread)
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)
    # model.load_state_dict(torch.load("SRCNNx2.pth"))

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    train_dataset = TrainDataset(args.train_dir, args.train_patch_size, args.upscale_factor, args.preupsampling)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.valid_dir, args.valid_patch_size, args.upscale_factor, args.preupsampling)
    eval_dataloader = DataLoader(dataset=eval_dataset, shuffle=False, batch_size=16)

    best_psnr = 0
    no_improve_count = 0
    early_stop_patience = 20

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=len(train_dataloader), desc=f'Train Epoch {epoch}/{args.num_epochs}') as t:
            for data in train_dataloader:
                inputs, labels = [d.to(device, non_blocking=True) for d in data]

                with autocast(device_type='cuda', dtype=torch.float16):
                    preds = model(inputs)

                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                t.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                t.update(1)

        if epoch % 1 == 0:
            model.eval()
            epoch_psnr = AverageMeter()

            with tqdm(total=len(eval_dataloader), desc=f'Valid Epoch {epoch}/{args.num_epochs}') as k:
                for data in eval_dataloader:
                    inputs, labels = [d.to(device, non_blocking=True) for d in data]

                    with torch.no_grad():
                        preds = model(inputs).clamp(0.0, 1.0)

                    psnr = calc_psnr(preds, labels)
                    epoch_psnr.update(psnr.item(), len(inputs))

                    k.set_postfix(PSNR=f'{epoch_psnr.avg:.6f}')
                    k.update(1)

            # Save the best model weights
            print(f'Eval PSNR: {epoch_psnr.avg:.2f}, Best PSNR: {best_psnr:.2f}')
            if epoch_psnr.avg > best_psnr:
                best_psnr = epoch_psnr.avg
                torch.save(model.state_dict(), args.save_weight_dir)
                print('/////////// The best weight was saved! ///////////')
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    print(
                        f'Early stopping at epoch {epoch} due to no PSNR improvement for {early_stop_patience} epochs.')
                    break

        # Clean up
        torch.cuda.empty_cache()

if __name__ == '__main__':
    train()
