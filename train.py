import pytorch_lightning as pl
import torch

from config import configs
from trainer import IceNetLightningModule, dataloader_train, dataloader_vali

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')

    # Initialize the Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=configs.num_epochs,
        precision='16-mixed',
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)]
    )

    # Initialize the model
    model = IceNetLightningModule()

    # Start training
    trainer.fit(model, dataloader_train, dataloader_vali)
