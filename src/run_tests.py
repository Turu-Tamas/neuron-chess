#%%
import torch.optim as optim
import torch
from model import ChessEncoder
from data_loader import make_dloaders, get_set_indices
from loss import multi_positive_loss
import lightning as L
from consts import *
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import seed_everything
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from tensorboard.plugins import projector
import shutil
from pathlib import Path
import os
from train import ChessTrainingModule
#%%
def main():
    torch.set_float32_matmul_precision('medium')
    seed_everything(RANDOM_SEED)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = make_dloaders()

    module = ChessTrainingModule()
    module.model = torch.load("weights/v13_temp0.07_emb32.pt", weights_only=False, map_location=device)
    trainer = L.Trainer(
        accelerator=device,
    )
    trainer.test(
        module,
        test_loader
    )

if __name__ == "__main__":
    main()
