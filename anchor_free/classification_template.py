from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torchvision import datasets, transforms
from torchmetrics import Accuracy


def set_seed(seed: int = 42) -> None:
    pl.seed_everything(seed, workers=True)


@dataclass
class TrainConfig:
    data_dir: str = "../input/imagenette/imagenette"
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 15
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.num_classes: Optional[int] = None

    @staticmethod
    def _transforms(phase: str):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if phase == "train":
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    def setup(self, stage: Optional[str] = None):
        self.train_ds = datasets.ImageFolder(
            root=f"{self.cfg.data_dir}/train",
            transform=self._transforms("train"),
        )
        self.val_ds = datasets.ImageFolder(
            root=f"{self.cfg.data_dir}/val",
            transform=self._transforms("val"),
        )
        self.test_ds = datasets.ImageFolder(
            root=f"{self.cfg.data_dir}/test",
            transform=self._transforms("val"),
        )
        self.num_classes = len(self.train_ds.classes)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.device == "cuda",
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.device == "cuda",
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.device == "cuda",
        )


class ImageClassifierModule(pl.LightningModule):
    def __init__(self, num_classes: int, cfg: TrainConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])  # avoid storing cfg dataclass directly
        self.cfg = cfg
        self.model = SimpleCNN(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(num_classes=num_classes)
        self.val_acc = Accuracy(num_classes=num_classes)
        self.test_acc = Accuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.train_acc.update(logits.softmax(dim=1).cpu(), y.cpu())
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.val_acc.update(logits.softmax(dim=1).cpu(), y.cpu())
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.test_acc.update(logits.softmax(dim=1).cpu(), y.cpu())
        self.log("test_loss", loss, prog_bar=False, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        self.test_acc.reset()


if __name__ == "__main__":
    cfg = TrainConfig()
    set_seed(cfg.seed)
    data = ImageDataModule(cfg)
    data.setup()
    model = ImageClassifierModule(num_classes=data.num_classes, cfg=cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.device,
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)


