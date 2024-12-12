import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from config import configs
from dataset.sic_dataset import SIC_dataset
from trainer import IceNetLightningModule


def create_parser():
    parser = argparse.ArgumentParser(description="OSI-SAF-RUA test")
    parser.add_argument(
        "-st",
        "--start_time",
        type=int,
        required=True,
        help="Starting time (six digits, YYYYMMDD)",
    )
    parser.add_argument(
        "-et",
        "--end_time",
        type=int,
        required=True,
        help="Ending time (six digits, YYYYMMDD)",
    )
    parser.add_argument(
        "-save",
        "--save_result",
        action="store_true",
        help="Whether to save the test results",
    )
    return parser


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    parser = create_parser()
    args = parser.parse_args()

    dataset_test = SIC_dataset(
        configs.data_paths,
        args.start_time,
        args.end_time,
        configs.input_gap,
        configs.input_length,
        configs.pred_shift,
        configs.pred_gap,
        configs.pred_length,
        samples_gap=1,
    )

    dataloader_test = DataLoader(dataset_test, shuffle=False)

    model = IceNetLightningModule()

    trainer = pl.Trainer()

    trainer.test(model, dataloaders=dataloader_test)
