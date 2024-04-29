import hydra
import lightning.pytorch as pl
import torch.utils.data
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

from pixel_rest_src.data_utils import SRGANDataModule
from pixel_rest_src.model import SRGANModel


@hydra.main(config_path="./", config_name="config", version_base="1.3")
def main(cfg: OmegaConf) -> None:
    text = (
        "==========================================\n"
        f"Training model with config:\n{cfg}"
        "\n=========================================="
    )
    print(text)

    data_module = SRGANDataModule(
        train_dataset_dir=cfg.train.data_path,
        val_dataset_dir=cfg.val.data_path,
        test_dataset_dir=None,
        train_shuffle=cfg.train.shuffle,
        val_shuffle=cfg.val.shuffle,
        test_shuffle=None,
        train_batch_size=cfg.train.batch_size,
        val_batch_size=cfg.val.batch_size,
        test_batch_size=None,
        num_workers=cfg.train.num_workers,
        crop_size=cfg.train.crop_size,
        upscale_factor=cfg.model.upscale_factor,
    )

    model = SRGANModel(cfg=cfg)

    mlf_logger = MLFlowLogger(
        experiment_name="SRGAN_logs", tracking_uri="0.0.0.0:5001"
    )  # from docker-compose config

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(),
    ]

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=callbacks,
        logger=mlf_logger,
        max_epochs=cfg.train.num_epochs,
        profiler="simple",
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
