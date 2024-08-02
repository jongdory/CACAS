import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn

import warnings
warnings.simplefilter("ignore")

from timm.models import create_model
from timm.utils import NativeScaler
from loss import MaskedDiceBCELoss, TotalLoss
from net.unet import Unet3D
from trainer import Trainer
from dataset import get_dataloader, CarotidDataset


def get_argparse():
    parser = argparse.ArgumentParser(description='Argparser')

    parser.add_argument('--train-path', type=str, default= './CosMos2022_Train')
    parser.add_argument('--test-path' , type=str, default= './CosMos2022_Test')
    parser.add_argument('--pred-path' , type=str, default= './results')
    parser.add_argument('--state-path', type=str, default= './weight')

    parser.add_argument('--model', type=str, default= "Unet3D", help="Unet3D")
    parser.add_argument('--batch-size', type=int, default= 2)
    parser.add_argument('--lr'   , type=float, default= 1e-2) # 1e-3
    parser.add_argument('--epoch', type=int, default= 800)
    parser.add_argument('--fold' , type=int, default= 0)
    parser.add_argument('--seed' , type=int, default= 55)

    parser.add_argument('--dp' , action='store_false')
    parser.set_defaults(dp=False)

    parser.add_argument('--mode' , type=str, default="total", help="seg, centd, cnst, centd_cnst, total")
    parser.add_argument('--name' , type=str, default="", help="")
    parser.add_argument('--nb-classes' , type=int, default=2, help="2 or 4")
    parser.add_argument('--r-lumen' , type=float, default=5.5, help="default 5.5")
    parser.add_argument('--r-wall' , type=float, default=7.5, help="default 7.5")

    args = parser.parse_args()

    return args


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    args = get_argparse()
    seed_everything(args.seed)
    print("mode:", args.mode ,"fold", args.fold)

    if args.mode == "seg":
        criterion = MaskedDiceBCELoss()
    else:
        criterion = TotalLoss(shape=(112,128), device="cuda", mode=args.mode)

    model = create_model(
        args.model,
        out_channels=2,
        pretrained=False).to('cuda')

    if args.dp:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    loss_scaler = NativeScaler()

    trainer = Trainer(net=model,
                  dataset=CarotidDataset,
                  criterion=criterion,
                  lr=args.lr,
                  accumulation_steps=args.batch_size * 4,
                  batch_size=args.batch_size,
                  fold=args.fold,
                  seed=args.seed,
                  num_epochs=args.epoch,
                  path=args.train_path,
                  loss_scaler=loss_scaler,
                  args=args)

    trainer.run()
    

if __name__ == "__main__":
    main()