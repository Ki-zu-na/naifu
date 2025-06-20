import safetensors
import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from typing import Callable

from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger

from diffusers import FlowMatchEulerDiscreteScheduler
from modules.sdxl_model import StableDiffusionModel
from lightning.pytorch.utilities.model_summary import ModelSummary

# ------------------------------------------------------------------------------------
# Reflow 训练脚本 (Flow-Matching → Reflow)
# 基本思路：
#   1. 数据集应提供配对 (noise_latent, image_latent)。若不存在，则默认以 x1 为图像，x0 随机采样噪声。
#   2. 训练目标 v_target = x1 - x0，与 Reflow 论文保持一致。
#   3. 使用均匀采样的 t ∈ (0,1) 构造插值 x_t = t * x1 + (1-t) * x0。
#   4. 损失函数为 MSE(v_pred, v_target)。
# ------------------------------------------------------------------------------------

def setup(fabric: pl.Fabric, config: OmegaConf):
    """与现有脚本保持一致的 setup 接口。"""
    model_path = config.trainer.model_path

    model = ReflowFineTune(
        model_path=model_path,
        config=config,
        device=fabric.device,
    )

    dataset_class = get_class(config.dataset.get("name", "data.AspectRatioDataset"))
    dataset = dataset_class(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        **config.dataset,
    )
    dataloader = dataset.init_dataloader()

    # -------- 优化器 & 调度器 --------
    params_to_optim = [{"params": model.model.parameters()}]
    if config.advanced.get("train_text_encoder_1"):
        lr = config.advanced.get("text_encoder_1_lr", config.optimizer.params.lr)
        params_to_optim.append({
            "params": model.conditioner.embedders[0].parameters(),
            "lr": lr,
        })
    if config.advanced.get("train_text_encoder_2"):
        lr = config.advanced.get("text_encoder_2_lr", config.optimizer.params.lr)
        params_to_optim.append({
            "params": model.conditioner.embedders[1].parameters(),
            "lr": lr,
        })

    optimizer = get_class(config.optimizer.name)(params_to_optim, **config.optimizer.params)
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(optimizer, **config.scheduler.params)

    # -------- checkpoint resume --------
    if config.trainer.get("resume"):
        latest_ckpt = get_latest_checkpoint(config.trainer.checkpoint_dir)
        remainder = {}
        if latest_ckpt:
            logger.info(f"Loading weights from {latest_ckpt}")
            remainder = sd = load_torch_file(ckpt=latest_ckpt, extract=False)
            if latest_ckpt.endswith(".safetensors"):
                remainder = safetensors.safe_open(latest_ckpt, "pt").metadata()
            model.load_state_dict(sd.get("state_dict", sd))
            config.global_step = remainder.get("global_step", 0)
            config.current_epoch = remainder.get("current_epoch", 0)

    model.first_stage_model.to(torch.float32)
    if fabric.is_global_zero and os.name != "nt":
        print(f"\n{ModelSummary(model, max_depth=1)}\n")

    # -------- fabric wrap --------
    if hasattr(fabric.strategy, "_deepspeed_engine"):
        model, optimizer = fabric.setup(model, optimizer)
        model.get_module = lambda: model
        model._deepspeed_engine = fabric.strategy._deepspeed_engine
    elif hasattr(fabric.strategy, "_fsdp_kwargs"):
        model, optimizer = fabric.setup(model, optimizer)
        model.get_module = lambda: model
        model._fsdp_engine = fabric.strategy
    else:
        model.model, optimizer = fabric.setup(model.model, optimizer)
        if config.advanced.get("train_text_encoder_1") or config.advanced.get("train_text_encoder_2"):
            model.conditioner = fabric.setup(model.conditioner)

    if hasattr(model, "mark_forward_method"):
        model.mark_forward_method("generate_samples")

    dataloader = fabric.setup_dataloaders(dataloader)

    model._fabric_wrapped = fabric
    model.model.requires_grad_(True)
    return model, dataset, dataloader, optimizer, scheduler


# ------------------------------------------------------------------------------------
class SupervisedFineTune(StableDiffusionModel):
    """在 Flow-Matching 权重基础上执行 Reflow 目标微调。"""

    def init_model(self):
        super().init_model()
        self.init_tag_loss_module()
        # 采样器仅用于后处理 draw sigma、但 Reflow 不用 α̇, σ̇。
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config({"num_train_timesteps": 1000})

    def init_tag_loss_module(self):
        if self.config.advanced.get("use_tag_loss", False):
            from modules.losses.tag_loss import TagLossModule

            def is_special_tag(tag: str) -> bool:
                return tag.startswith((
                    "artist:", "character:", "rating:", "style:", "copyright:", "year "))

            self.tag_loss_module = TagLossModule(
                check_fn=is_special_tag,
                alpha=self.config.advanced.get("tag_loss_alpha", 0.2),
                beta=self.config.advanced.get("tag_loss_beta", 0.99),
                strength=self.config.advanced.get("tag_loss_strength", 1.0),
                tag_rewards=self.config.advanced.get("tag_rewards", {}),
            )

    # ------------------ forward ------------------
    def forward(self, batch):
        """batch 必须包含
            - pixels : 图像或其 latent 表示 (x1)
            - noise  : 与图像配对的 gaussian latent (x0)。若缺失则自动随机。"""
        # ---- 准备两端 latent ----
        if batch.get("is_latent", False):
            x1 = self._normliaze(batch["pixels"])          # 已经是 latent
        else:
            self.first_stage_model.to(self.target_device)
            x1 = self.encode_first_stage(batch["pixels"].to(self.first_stage_model.dtype))

        x0 = batch.get("noise")
        if x0 is None:
            x0 = torch.randn_like(x1)
        else:
            x0 = x0.to(x1.device, dtype=x1.dtype)

        # ---- 条件编码 ----
        cond = self.encode_batch(batch)
        model_dtype = next(self.model.parameters()).dtype
        cond = {k: v.to(model_dtype) for k, v in cond.items()}

        # ---- 采样随机 t, 构造插值 ----
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=self.target_device)
        t_view = t.view(-1, 1, 1, 1)
        x_t = t_view * x1 + (1 - t_view) * x0

        # ---- 目标 & 预测 ----
        v_target = x1 - x0                       # 常数，不随 t 变
        timesteps = (t * 999).long()            # 整数时间步编码
        model_pred = self.model(x_t.to(model_dtype), timesteps, cond)
        base_loss = torch.mean(((model_pred.float() - v_target.float())**2).reshape(bsz, -1), 1)

        # ---- 可选 TagLoss ----
        if hasattr(self, "tag_loss_module"):
            self.tag_loss_module.global_step = self.global_step
            weights = self.tag_loss_module.calculate_loss_weights(batch["prompts"], base_loss.detach())
            loss = (base_loss * weights).mean()
        else:
            loss = base_loss.mean()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"NaN/Inf loss detected: {loss.item()}")
            raise RuntimeError("NaN or Inf encountered in loss, aborting.")
        return loss 