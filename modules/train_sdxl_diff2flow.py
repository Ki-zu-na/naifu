import safetensors
import torch
import os, math
import lightning as pl
from typing import Callable
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger

from diffusers import  DDPMScheduler
from modules.sdxl_model import StableDiffusionModel
from modules.scheduler_utils import apply_snr_weight,apply_zero_terminal_snr
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
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear", 
            num_train_timesteps=1000,
            clip_sample=False, # Diff2Flow 通常不裁剪
        )
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.target_device)

        apply_zero_terminal_snr(self.noise_scheduler)
        
        logger.info("DDPMScheduler for Diff2Flow initialized.")
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

        cond = self.encode_batch(batch)
        model_dtype = next(self.model.parameters()).dtype
        cond = {k: v.to(model_dtype) for k, v in cond.items()}
        
        if not hasattr(self, 'fm_t_map'):
            self.fm_t_map = get_fm_t_to_dm_t_map(self.noise_scheduler)
        # --- 训练循环开始 ---
        bsz = latents.shape[0]
        device = latents.device

        # 1. 在FM空间进行操作
        # 1.1 准备 x0 (噪声) 和 x1 (数据)
        x1 = latents
        x0 = torch.randn_like(x1)

        # 1.2 采样连续时间 t_fm in [0, 1]
        t_fm = torch.rand(bsz, device=device, dtype=latents.dtype)

        # 1.3 计算FM的插值 xt_fm 和目标速度场 ut
        # 根据论文，t_fm 是权重，但通常实现时会reshape
        t_fm_reshaped = t_fm.view(-1, 1, 1, 1)
        xt_fm = t_fm_reshaped * x1 + (1.0 - t_fm_reshaped) * x0
        ut_target = x1 - x0 # 这是我们的真实目标速度场

        # 2. Diff2Flow: 将FM变量转换为DM变量
        # 2.1 将 (xt_fm, t_fm) 转换为 (xt_dm, t_dm)
        xt_dm, t_dm = convert_fm_xt_to_dm_xt(xt_fm, t_fm, self.noise_scheduler, self.fm_t_map)

        # 3. 模型预测 (在DM空间)
        # 模型输入转换后的 xt_dm 和 t_dm
        # 注意：t_dm是连续的浮点数，模型需要能处理它。
        # 幸运的是，扩散模型的时间编码（如Sinusoidal-Position-Embedding）天然支持浮点数输入。
        v_pred_dm = self.model(xt_dm.to(model_dtype), t_dm, cond)

        # 4. Diff2Flow: 将DM预测转换回FM速度场
        # 这里的t_dm需要取整以从scheduler中索引alpha/sigma，但v_pred_dm是基于连续t_dm预测的
        vector_pred = get_vector_field_from_v(v_pred_dm, xt_dm, t_dm, self.noise_scheduler)
        # 官方代码在 get_vector_field_from_v 内部处理了连续t_dm的插值，
        # 这里为了简化，我们直接用连续 t_dm 去索引，但需要注意精度。
        # 更严谨的做法是在get_vector_field_from_v内部也对alpha_t和sigma_t进行插值。
        # 但为了保持与你原始代码的相似性，我们先用取整的方式。

        # 5. 计算损失 (在FM空间)
        # 比较模型预测的速度场 vector_pred 和真实目标 ut_target
        base_loss = torch.mean(((vector_pred.float() - ut_target.float()) ** 2).reshape(bsz, -1), 1)

        # --- 训练循环结束 ---

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

# --- 工具函数：从官方代码中借鉴和简化 ---

def get_fm_t_to_dm_t_map(scheduler):
    """
    预先计算从 FM 时间 t_fm (0->1) 到 DM 时间 t_dm (1000->0) 的映射。
    这是论文 Eq. (11) 的实现。
    """
    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
    
    # 论文中的 f(t_DM) = alpha_t / (alpha_t + sigma_t)
    # 这就是 t_FM 的值。注意这个值是从1到0.5递减的。
    # 为了映射到 [0, 1]，通常会做一些调整，但官方代码的实现更直接，
    # 它用 searchsorted 在这个递减的序列中查找。
    # 我们这里也遵循这个逻辑。
    # 官方代码中还包含了一个为 zero-terminal-snr 做的 full_schedule,
    # 这里为了简化，我们直接使用 alphas_cumprod
    fm_t_map = sqrt_alphas_cumprod / (sqrt_alphas_cumprod + sqrt_one_minus_alphas_cumprod)
    return fm_t_map.cpu() # 将其移动到CPU，作为查找表

def convert_fm_t_to_dm_t(t_fm, fm_t_map, num_train_timesteps):
    """
    将一批 FM 连续时间 t_fm [0,1] 转换为 DM 连续时间 t_dm。
    这是论文 Eq. (12) 的逆向查找和插值过程。
    """
    orig_device = t_fm.device
    t_cpu = t_fm.cpu()

    # fm_t_map 是一个从 t_dm=0 到 t_dm=999 的递减序列
    # 我们将其翻转，得到一个 t_dm 从 999->0 的递增序列
    reversed_map = torch.flip(fm_t_map, [0])
    
    # 在这个递增序列中查找 t_fm 的位置
    # searchsorted 要求被搜索的张量（reversed_map）是单调不减的
    # t_cpu 也是单调不减的，所以可以直接搜索
    right_indices = torch.searchsorted(reversed_map, t_cpu, right=True)
    left_indices = right_indices - 1
    
    # 处理边界情况
    left_indices = torch.clamp(left_indices, 0, len(reversed_map) - 2)
    right_indices = torch.clamp(right_indices, 1, len(reversed_map) - 1)

    right_values = reversed_map[right_indices]
    left_values = reversed_map[left_indices]

    # 线性插值
    interp_weights = (t_cpu - left_values) / (right_values - left_values)
    interp_weights = torch.nan_to_num(interp_weights, 0.0) # 防止除以0
    
    # 得到的 dm_t_reversed 是 t_dm 从 999->0 的连续浮点数索引
    dm_t_reversed = left_indices.float() + interp_weights * (right_indices.float() - left_indices.float())

    # 将反向的 t_dm (0->999) 转换回正向的 t_dm (999->0)
    dm_t = (num_train_timesteps - 1) - dm_t_reversed
    return dm_t.to(orig_device)


def convert_fm_xt_to_dm_xt(xt_fm, t_fm, scheduler, fm_t_map):
    """
    将 FM 空间的带噪样本 xt_fm 转换为 DM 空间的带噪样本 xt_dm。
    这是论文 Eq. (13) 的实现。
    """
    device = xt_fm.device
    sqrt_alphas_cumprod = scheduler.alphas_cumprod.to(device).sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - scheduler.alphas_cumprod.to(device)).sqrt()
    scale = sqrt_alphas_cumprod + sqrt_one_minus_alphas_cumprod
    
    # 为了获取对应 t_fm 的 scale, 我们需要先得到 t_dm
    t_dm = convert_fm_t_to_dm_t(t_fm, fm_t_map, scheduler.config.num_train_timesteps)
    
    # 在离散的 scale 值之间进行线性插值
    t_dm_floor = torch.floor(t_dm).long()
    t_dm_ceil = torch.ceil(t_dm).long()

    # 处理边界
    t_dm_floor = torch.clamp(t_dm_floor, 0, len(scale) - 1)
    t_dm_ceil = torch.clamp(t_dm_ceil, 0, len(scale) - 1)
    
    scale_floor = scale[t_dm_floor]
    scale_ceil = scale[t_dm_ceil]
    
    interp_weights = (t_dm - t_dm_floor.float()).view(-1, 1, 1, 1)
    
    scale_t = scale_floor.view(-1, 1, 1, 1) * (1 - interp_weights) + scale_ceil.view(-1, 1, 1, 1) * interp_weights
    
    xt_dm = xt_fm * scale_t
    return xt_dm, t_dm


def get_vector_field_from_v(v_pred, xt_dm, t_dm, scheduler):
    """
    从模型的 v-prediction 输出计算 FM 的速度场。
    这等同于你之前的实现，只是封装成了一个函数。
    """
    device = v_pred.device
    num_train_timesteps = scheduler.config.num_train_timesteps
    
    # --- 关键修改：在这里 clamp 索引 ---
    # 确保 t_dm 在 [0, 999] 的范围内
    t_dm_clamped = torch.clamp(t_dm, 0.0, num_train_timesteps - 1.0)
    
    # 使用 clamp 后的 t_dm 进行索引
    indices = t_dm_clamped.long()
    alphas_cumprod_device = scheduler.alphas_cumprod.to(device)
    alpha_t = alphas_cumprod_device[indices].sqrt().view(-1, 1, 1, 1)
    sigma_t = (1.0 - alphas_cumprod_device[indices]).sqrt().view(-1, 1, 1, 1)

    # eps_pred = alpha_t * v_pred + sigma_t * xt_dm
    # z_pred   = alpha_t * xt_dm - sigma_t * v_pred
    # vector_pred = z_pred - eps_pred
    # 简化后是：
    vector_pred = (alpha_t - sigma_t) * xt_dm - (alpha_t + sigma_t) * v_pred
    return vector_pred