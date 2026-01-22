import torch.optim as optim
import torch
from model import ChessEncoder
from data_loader import make_dloaders
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

class ChessTrainingModule(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ChessEncoder(ENCODER_EMBEDDING_DIM)

    def forward(self, states):
        input_states = states.view(-1, *INPUT_SIZE)
        return self.model(input_states)

    def training_step(self, batch, batch_idx):
        embeddings = self(batch)
        loss = multi_positive_loss(embeddings, group_size=POSITIONS_PER_GAME, temperature=TEMPERATURE)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=LR)
        return optimizer

    def validation_step(self, states):
        input_states = states.view(-1, *INPUT_SIZE)
        embeddings = self(input_states)
        loss = multi_positive_loss(embeddings, group_size=POSITIONS_PER_GAME, temperature=TEMPERATURE)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_test_start(self):
        self.test_embeddings = []

    def test_step(self, batch):
        embeddings = self(batch)
        self.test_embeddings.append(embeddings)
        loss = multi_positive_loss(embeddings, POSITIONS_PER_GAME, TEMPERATURE)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def on_test_end(self):
        embeddings_tensor = torch.cat(self.test_embeddings)[:POSITIONS_PER_GAME * VISUALIZATION_GAME_COUNT]
        writer = self.logger.experiment
        writer.add_embedding(
            mat=embeddings_tensor,
            metadata=np.repeat(np.arange(len(embeddings_tensor) // POSITIONS_PER_GAME), POSITIONS_PER_GAME),
            tag="test_embeddings"
        )
        self.test_embeddings.clear()

def main():
    torch.set_float32_matmul_precision('medium')
    seed_everything(RANDOM_SEED)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = make_dloaders()

    module = ChessTrainingModule()
    trainer = L.Trainer(
        accelerator=device,
        callbacks=[EarlyStopping("val_loss", min_delta=1e-3, patience=3)],
        max_epochs=-1
    )
    trainer.fit(
        module, train_dataloaders=train_loader, val_dataloaders=val_loader,
    )
    trainer.test()

if __name__ == "__main__":
    main()