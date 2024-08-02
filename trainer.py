import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn

from torch.optim import AdamW, RAdam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, ChainedScheduler
from time import time
from IPython.display import clear_output
from dataset import get_dataloader
from loss import Meter


def maybe_mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


class Trainer:
    """
    Factory for training proccess.
    Args:
        display_plot: if True - plot train history after each epoch.
        net: neural network for mask prediction.
        criterion: factory for calculating objective loss.
        optimizer: optimizer for weights updating.
        phases: list with train and validation phases.
        dataloaders: dict with data loaders for train and val phases.
        meter: factory for storing and updating metrics.
        batch_size: data batch size for one step weights updating.
        num_epochs: num weights updation for all data.
        accumulation_steps: the number of steps after which the optimization step can be taken.
        lr: learning rate for optimizer.
        scheduler: scheduler for control learning rate.
        losses: dict for storing lists with losses for each phase.
        jaccard_scores: dict for storing lists with jaccard scores for each phase.
        dice_scores: dict for storing lists with dice scores for each phase.
    """
    def __init__(self,
                 net: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 criterion: nn.Module,
                 lr: float,
                 accumulation_steps: int,
                 batch_size: int,
                 fold: int,
                 seed: int,
                 num_epochs: int,
                 path: str,
                 display_plot: bool = True,
                 loss_scaler = None,
                 args = None
                ):

        """Initialization."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.display_plot = display_plot
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = AdamW(self.net.parameters(), lr=lr, betas=(.95, 0.999), eps=1e-6, weight_decay=0)
        self.scheduler_exp = ExponentialLR(self.optimizer, gamma=0.995)
        self.scheduler_red = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, threshold = 1e-6, patience=30, eps=1e-3, verbose=True)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "val"]
        self.num_epochs = num_epochs
        self.loss_scaler = loss_scaler
        self.args = args

        self.dataloaders = {
            phase: get_dataloader(
                dataset = dataset,
                path = path,
                phase = phase,
                fold = fold,
                seed = seed,
                batch_size = batch_size,
                num_workers = 8,
            )
            for phase in self.phases
        }
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.bce_scores = {phase: [] for phase in self.phases}
        self.cent_scores = {phase: [] for phase in self.phases}
        self.cnst_scores = {phase: [] for phase in self.phases}
        self.bcedice_scores = {phase: [] for phase in self.phases}
         

    def _compute_loss_and_outputs(self,
                                  images: torch.Tensor,
                                  targets: torch.Tensor,
                                  centerlines: torch.Tensor,
                                  weightmats: torch.Tensor,
                                  radiuses: np.array,
                                  epoch: int):

        images  = images.to(self.device)
        targets = targets.to(self.device)
        centerlines = centerlines.to(self.device)
        weightmats  = weightmats.to(self.device)

        logits = self.net(images)
        if self.args.mode == "seg":
            loss = self.criterion(logits, targets, weightmats)
            loss_bw = loss
        else:
            loss_bw, loss = self.criterion(logits, targets, weightmats, centerlines, radiuses, epoch)

        return loss_bw, loss, logits
        

    def _do_epoch(self, epoch: int, phase: str, optimizer, loss_scaler, radiuses):
        torch.autograd.set_detect_anomaly(True)
        st_time = time()

        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        optimizer.zero_grad()
        
        for itr, data_batch in enumerate(dataloader):
            images, targets = data_batch['image'], data_batch['mask']
            weightmats, centerlines = data_batch['weightmat'], data_batch['centerline']

            with torch.autograd.detect_anomaly():
                loss_bw, loss, logits = self._compute_loss_and_outputs(images, targets, centerlines, weightmats, radiuses=radiuses, epoch=epoch)
                loss = loss / self.accumulation_steps
                loss_bw = loss_bw / self.accumulation_steps
                if phase == "train":
                    loss_bw.backward()
                    if (itr + 1) % self.accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

            running_loss += loss.item()

            meter.update(logits.detach().type(torch.FloatTensor),
                         targets.detach(),
                         weightmats.detach(),
                         centerlines.detach(),
                         radiuses=radiuses
                        )
        
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        
        epoch_dice, epoch_bce, epoch_cent, epoch_cnst = meter.get_metrics()

        epoch_bcedice = (1 - epoch_dice) + epoch_bce

        if phase == "train":
            print(f"{phase} epoch: {epoch+1} / {self.num_epochs}")
        print(f" dice: {epoch_dice:.4f} |  bce: { epoch_bce:.4f} |    loss: {epoch_loss:.4f} ")
        print(f" cent: {epoch_cent:.4f} | cnst: {epoch_cnst:.4f} | bcedice: {epoch_bcedice:.4f} ")
        print(f"learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"This epoch took {time()-st_time:.2f}s \n")
        
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.bce_scores[phase].append(epoch_bce)
        self.cent_scores[phase].append(epoch_cent)
        self.cnst_scores[phase].append(epoch_cnst)
        self.bcedice_scores[phase].append(epoch_bcedice)

        return epoch_loss


    def run(self):
        mode_name = self.args.mode
        radiuses = np.array([[self.args.r_lumen, self.args.r_wall], [self.args.r_lumen - 2, self.args.r_wall - 2]])

        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train", self.optimizer, self.loss_scaler, radiuses)
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val", self.optimizer, self.loss_scaler, radiuses)
                self.scheduler_exp.step()
                self.scheduler_red.step(val_loss)
            if self.display_plot:
                self._plot_train_history()
                
            if  val_loss < self.best_loss:
                print(f"----------------------\n Saved new checkpoint\n----------------------")
                self.best_loss = val_loss
                maybe_mkdir(f"{self.args.state_path}/{self.args.model}_{mode_name}_{self.args.name}_fold{self.args.fold}")
                torch.save(self.net.state_dict(), f"{self.args.state_path}/{self.args.model}_{mode_name}_{self.args.name}_fold{self.args.fold}/best_model.pth")
            print()
        self._save_train_history()
            

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.bce_scores, self.cent_scores, self.cnst_scores, self.bcedice_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,
            
            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]} 
            """, 
                  
            f"""
            train bce score {self.bce_scores['train'][-1]}
            val bce score {self.bce_scores['val'][-1]}
            """,

            f"""
            train cent score {self.cent_scores['train'][-1]}
            val cent score {self.cent_scores['val'][-1]}
            """,

            f"""
            train cnst score {self.cnst_scores['train'][-1]}
            val cnst score {self.cnst_scores['val'][-1]}
            """,

            f"""
            train bcedice score {self.bcedice_scores['train'][-1]}
            val bcedice score {self.bcedice_scores['val'][-1]}
            """,
        ]
        
        clear_output(True)

        fig, axes = plt.subplots(6, 1, figsize=(8, 24))
        for i, ax in enumerate(axes):
            ax.plot(data[i]['val'], c=colors[0], label="val")
            ax.plot(data[i]['train'], c=colors[-1], label="train")
            ax.set_title(labels[i])
            ax.legend(loc="upper right")
            
        plt.tight_layout()
        mode_name = self.args.mode
        maybe_mkdir(f"./log/{self.args.model}_{mode_name}_{self.args.name}_fold{self.args.fold}")
        plt.savefig(f"./log/{self.args.model}_{mode_name}_{self.args.name}_fold{self.args.fold}/loss.png")
            

    def load_predtrain_model(self,
                             state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")
        

    def _save_train_history(self):
        """writing model weights and training logs to files."""
        mode_name = self.args.mode
        maybe_mkdir(f"{self.args.state_path}/{self.args.model}_{mode_name}_{self.args.name}_fold{self.args.fold}")
        torch.save(self.net.state_dict(),
                   f"{self.args.state_path}/{self.args.model}_{mode_name}_{self.args.name}_fold{self.args.fold}/last_model.pth")

        logs_ = [self.losses, self.dice_scores, self.bce_scores, self.cent_scores, self.cnst_scores, self.bcedice_scores]
        log_names_ = ["_loss", "_dice", "_bce", "_cent", "_cnst", "_bcedice"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                         for key in logs_[i]]
        log_names = [key+log_names_[i] 
                     for i in list(range(len(logs_))) 
                     for key in logs_[i]
                    ]
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv(f"./log/{self.args.model}_{mode_name}_{self.args.name}_fold{self.args.fold}/train_log.csv", index=False)