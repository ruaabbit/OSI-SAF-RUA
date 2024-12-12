from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.amp import autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import OneCycleLR, LRScheduler
from torch.utils.data import DataLoader

from config import configs
from dataset.sic_dataset import SIC_dataset
from utils import IceNet, mse_func, rmse_func, mae_func, nse_func, PSNR_func, BACC_func

dataset_train = SIC_dataset(
    configs.data_paths,
    configs.train_period[0],
    configs.train_period[1],
    configs.input_gap,
    configs.input_length,
    configs.pred_shift,
    configs.pred_gap,
    configs.pred_length,
    samples_gap=1,
)

dataset_vali = SIC_dataset(
    configs.data_paths,
    configs.eval_period[0],
    configs.eval_period[1],
    configs.input_gap,
    configs.input_length,
    configs.pred_shift,
    configs.pred_gap,
    configs.pred_length,
    samples_gap=1,
)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=configs.batch_size,
    num_workers=configs.num_workers,
    shuffle=True,
)

dataloader_vali = DataLoader(
    dataset_vali,
    batch_size=configs.batch_size_vali,
    num_workers=configs.num_workers,
    shuffle=False,
)


class IceNetLightningModule(pl.LightningModule):
    """
    用于海冰浓度预测的 PyTorch Lightning 模块。

    该模块处理 IceNet 模型的训练、验证和测试，并记录各种性能指标。
    """

    def __init__(self, metrics_to_log: Optional[List[str]] = None):
        """
        初始化 IceNetLightningModule。

        参数:
            metrics_to_log (Optional[List[str]], 可选): 要记录的指标列表。
                默认为所有可用指标。
        """
        super().__init__()

        # 如果未指定，则使用默认指标
        self.metrics_to_log = metrics_to_log or [
            'mse', 'rmse', 'mae', 'nse', 'PSNR', 'BACC'
        ]

        # 将指标名称映射到其计算函数
        self.metric_funcs = {
            'mse': mse_func,
            'rmse': rmse_func,
            'mae': mae_func,
            'nse': nse_func,
            'PSNR': PSNR_func,
            'BACC': BACC_func
        }

        # 加载北极掩码
        self.arctic_mask = torch.from_numpy(np.load("data/AMAP_mask.npy"))
        self.network = IceNet()

        # 保存所有配置作为超参数
        self.save_hyperparameters(configs.__dict__)

    def _compute_and_log_metrics(
            self,
            sic_pred: torch.Tensor,
            targets: torch.Tensor,
            mask: torch.Tensor,
            stage: str
    ) -> torch.Tensor:
        """
        计算并记录性能指标。

        参数:
            sic_pred (torch.Tensor): 预测的海冰浓度
            targets (torch.Tensor): 目标海冰浓度
            mask (torch.Tensor): 北极掩码
            stage (str): 当前阶段（'train'、'val' 或 'test'）

        返回:
            torch.Tensor: 损失值
        """
        # 计算指标
        metrics = {}
        for metric_name in self.metrics_to_log:
            metric_value = self.metric_funcs[metric_name](sic_pred, targets, mask)
            metrics[f'{stage}_{metric_name}'] = metric_value
            self.log(f'{stage}_{metric_name}', metric_value,
                     on_step=True, on_epoch=True, prog_bar=True)

        return metrics.get(f'{stage}_loss', torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型的前向传播。

        参数:
            inputs (torch.Tensor): 输入数据
            targets (torch.Tensor): 目标数据

        返回:
            Tuple[torch.Tensor, torch.Tensor]: 预测的海冰浓度和损失
        """
        with autocast(device_type="cuda"):
            sic_pred, loss = self.network(inputs, targets)
        return sic_pred, loss

    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        单个批次的训练步骤。

        参数:
            batch (List[torch.Tensor]): 输入数据批次
            batch_idx (int): 批次索引

        返回:
            torch.Tensor: 损失值
        """
        inputs, targets = batch
        inputs = inputs.float()
        targets = targets.float()
        mask = self.arctic_mask.float().to(self.device)

        sic_pred, loss = self(inputs, targets)

        # 记录训练指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return self._compute_and_log_metrics(sic_pred, targets, mask, 'train')

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        单个批次的验证步骤。

        参数:
            batch (List[torch.Tensor]): 输入数据批次
            batch_idx (int): 批次索引

        返回:
            torch.Tensor: 损失值
        """
        inputs, targets = batch
        inputs = inputs.float()
        targets = targets.float()
        mask = self.arctic_mask.float().to(self.device)

        sic_pred, loss = self(inputs, targets)

        # 记录验证指标
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return self._compute_and_log_metrics(sic_pred, targets, mask, 'val')

    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        单个批次的测试步骤。

        参数:
            batch (List[torch.Tensor]): 输入数据批次
            batch_idx (int): 批次索引

        返回:
            torch.Tensor: 损失值
        """
        inputs, targets = batch
        inputs = inputs.float()
        targets = targets.float()
        mask = self.arctic_mask.float().to(self.device)

        sic_pred, loss = self(inputs, targets)

        # 记录测试指标
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return self._compute_and_log_metrics(sic_pred, targets, mask, 'test')

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        """
        配置优化器和学习率调度器。

        返回:
            Tuple[List[Optimizer], List[Scheduler]]: 优化器和学习率调度器
        """
        optimizer = AdamW(
            self.network.parameters(),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            epochs=configs.num_epochs,
            steps_per_epoch=len(dataloader_train),
            max_lr=configs.lr,
        )
        return [optimizer], [lr_scheduler]
