import safetensors
import torch
import os, math
import lightning as pl
from typing import Callable
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger
from torch.autograd.functional import jvp

from diffusers import  FlowMatchEulerDiscreteScheduler
from modules.sdxl_model import StableDiffusionModel
from modules.scheduler_utils import apply_snr_weight
from lightning.pytorch.utilities.model_summary import ModelSummary


def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    model_path = config.trainer.model_path

    model = SupervisedFineTune(
        model_path=model_path, 
        config=config, 
        device=fabric.device
    )
    dataset_class = get_class(config.dataset.get("name", "data.AspectRatioDataset"))
    dataset = dataset_class(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        **config.dataset,
    )
    dataloader = dataset.init_dataloader()
    
    params_to_optim = [{'params': model.model.parameters()}]
    if config.advanced.get("train_text_encoder_1"):
        lr = config.advanced.get("text_encoder_1_lr", config.optimizer.params.lr)
        params_to_optim.append(
            {"params": model.conditioner.embedders[0].parameters(), "lr": lr}
        )
        
    if config.advanced.get("train_text_encoder_2"):
        lr = config.advanced.get("text_encoder_2_lr", config.optimizer.params.lr)
        params_to_optim.append(
            {"params": model.conditioner.embedders[1].parameters(), "lr": lr}
        )

    optim_param = config.optimizer.params
    optimizer = get_class(config.optimizer.name)(
        params_to_optim, **optim_param
    )
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )
    
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
        model.mark_forward_method('generate_samples')            
        
    dataloader = fabric.setup_dataloaders(dataloader)
        
    # set here; 
    model._fabric_wrapped = fabric
    model.model.requires_grad_(True)
    return model, dataset, dataloader, optimizer, scheduler


class SupervisedFineTune(StableDiffusionModel):
    def init_model(self):
        super().init_model()
        self.init_tag_loss_module()

    def init_tag_loss_module(self):
        # 初始化tag loss模块
        if self.config.advanced.get("use_tag_loss", False):
            from modules.losses.tag_loss import TagLossModule
            
            def is_special_tag(tag: str) -> bool:
                return tag.startswith(("artist:", "character:", "rating:", "style:", "copyright:","year "))
            
            self.tag_loss_module = TagLossModule(
                check_fn=is_special_tag,
                alpha=self.config.advanced.get("tag_loss_alpha", 0.2),
                beta=self.config.advanced.get("tag_loss_beta", 0.99),
                strength=self.config.advanced.get("tag_loss_strength", 1.0),
                tag_rewards=self.config.advanced.get("tag_rewards", {})
            )

    def forward(self, batch):
        advanced = self.config.get("advanced", {})
        if not batch["is_latent"]:
            self.first_stage_model.to(self.target_device)
            latents = self.encode_first_stage(batch["pixels"].to(self.first_stage_model.dtype))
            if torch.any(torch.isnan(latents)):
                logger.info("NaN found in latents, replacing with zeros")
                latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
        else:
            self.first_stage_model.cpu()
            latents = self._normliaze(batch["pixels"])

        # --- 1. 前置准备 (Setup) ---
        # latents -> x1 (目标，真实的清晰数据)
        # noise -> x0 (源头，纯噪声)
        cond = self.encode_batch(batch)
        model_dtype = next(self.model.parameters()).dtype
        cond = {k: v.to(model_dtype) for k, v in cond.items()}

        x1 = latents
        x0 = torch.randn_like(x1, device=x1.device) # 重新生成噪声以确保与 x1 匹配

        bsz = x1.shape[0]

        # --- 2. 流匹配核心改造 (Flow Matching Core Refactor) ---

        t = torch.rand(bsz, device=x1.device).view(-1, 1, 1, 1)

        # 核心概念1: 定义数据点 xt 在从 x1 到 x0 的直线路径上的位置
        # t=0 时, xt = x1 (真实数据)
        # t=1 时, xt = x0 (纯噪声)
        xt = (1 - t) * x1 + t * x0

        # 核心概念2: 定义这条路径上任意点 xt 应该流向的 "真实" 速度/方向向量
        # 对于直线路径，这个向量场是恒定的，即从起点指向终点
        vt = x0 - x1

        # --- 3. 使用 JVP 计算动态损失 (Dynamic Loss with JVP) ---

        # 定义一个包装了模型的匿名函数，用于 JVP 计算
        # 它接收 xt 和 t 作为输入
        jvp_fn = lambda _xt, _t: self.model(_xt.to(model_dtype), _t.squeeze() * 1000, cond)
        # 注意: _t.squeeze() * 1000 是为了模拟你之前的 timesteps 输入，你可能需要根据你的模型调整

        # --------- 使用 torch.autograd.functional.jvp ----------
        # jvp_fn 接收一个元组作为输入
        wrapped_fn = lambda _xt, _t: jvp_fn(_xt, _t)
        
        # JVP 计算: create_graph=True 允许我们通过 JVP 的结果进行反向传播
        # v 代表了输入 (xt, t) 的变化方向，对于 xt 是 vt，对于 t 则是 1
        u, dudt = jvp(
            wrapped_fn,
            (xt, t),
            v=(vt, torch.ones_like(t)),
            create_graph=True
        )
        
        u = u.to(model_dtype)
        dudt = dudt.to(model_dtype)

        # --- 4. 定义新的损失函数 ---
        # 损失由两部分组成:
        # 1. "方向损失": 确保模型输出 u 的方向和大小与真实向量场 vt 一致 (和 v-prediction 类似)
        loss_direction = torch.mean(((u - vt) ** 2).reshape(bsz, -1), 1)

        # 2. "稳定损失" (JVP项): 惩罚 dudt，促使模型预测在流线上保持稳定，理想状态下 dudt=0
        loss_stability = torch.mean((dudt ** 2).reshape(bsz, -1), 1)

        # 超参数 alpha，用于平衡两个损失项的重要性
        # alpha 越大，对模型预测的稳定性要求越高
        alpha = 0.15

        # 最终损失
        final_loss = loss_direction + alpha * loss_stability

        # (可选) 你可以只使用一个，但组合起来通常效果更好
        # final_loss = loss_direction

        base_loss = final_loss # 将最终损失赋值给你的 base_loss 变量


        if hasattr(self, "tag_loss_module"):
            self.tag_loss_module.global_step = self.global_step
            weights = self.tag_loss_module.calculate_loss_weights(
                batch["prompts"],
                base_loss.detach()
            )

            if hasattr(self, "log_dict"):
                log_dict = {
                    "train/tag_loss_weight": weights.mean().item(),
                    "train/weighted_loss": (base_loss * weights).mean().item(),
                    "train/max_weight": weights.max().item(),
                    "train/min_weight": weights.min().item(),
                    "train/special_tags_count": sum(1 for prompt in batch["prompts"]
                                                    for tag in prompt.split(",")
                                                    if self.tag_loss_module.check_fn(tag.strip()))
                }
                self.log_dict(log_dict)
                self.tag_loss_metrics = log_dict

            loss = (base_loss * weights).mean()
        else:
            loss = base_loss.mean()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"Error: NaN or Inf loss encountered! Loss value: {loss.item()}")
            raise RuntimeError("NaN or Inf loss detected, stopping training.")
        
        return loss
