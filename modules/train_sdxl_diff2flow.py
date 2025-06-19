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
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear", 
            num_train_timesteps=1000,
            clip_sample=False, # Diff2Flow 通常不裁剪
        )
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.target_device)
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

        bsz = latents.shape[0]
        noise = torch.randn_like(latents, device=latents.device)

        # 1. 从调度器的整个时间范围内随机采样离散时间步
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        
        # 2. 获取对齐扩散路径所需的 alpha, sigma 及其导数
        alpha_t, sigma_t, alpha_dot_t, sigma_dot_t = get_diffusion_schedule_properties(
            self.noise_scheduler, 
            timesteps, 
            device=latents.device,
            dtype=latents.dtype
        )

        # 3. 使用扩散模型的 schedule 构造带噪样本 x_t
        #    这确保了插值路径与预训练模型一致
        noisy_latents = alpha_t * latents + sigma_t * noise
        
        # 4. 计算真实的速度目标 v_target = d(x_t)/dt
        #    v_target = d(alpha_t * latents + sigma_t * noise) / dt
        #             = (d(alpha_t)/dt) * latents + (d(sigma_t)/dt) * noise
        #             = alpha_dot_t * latents + sigma_dot_t * noise
        target_v = alpha_dot_t * latents + sigma_dot_t * noise

        # --- 替换结束 ---

        # 模型预测速度场 v_pred
        # 注意：timesteps 传递给U-Net，它内部会处理
        model_pred = self.model(noisy_latents.to(model_dtype), timesteps, cond)
        base_loss = torch.mean(((model_pred.float() - target_v.float()) ** 2).reshape(latents.shape[0], -1), 1)


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

def get_sigmas(sch, timesteps, n_dim=4, dtype=torch.float32, device="cuda:0"):
    sigmas = sch.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = sch.timesteps.to(device)
    timesteps = timesteps.to(device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_diffusion_schedule_properties(scheduler: DDPMScheduler, timesteps: torch.Tensor, device: torch.device, dtype: torch.dtype):
    """
    计算给定时间步的 alpha, sigma 及其导数。
    这是 Diff2Flow 的核心，用于计算真实的速度目标。
    """
    # 将离散的 timesteps (0-999) 映射到连续时间 t (0-1)
    # 我们假设 scheduler.timesteps 是 [999, 998, ..., 0]
    continuous_t = (scheduler.config.num_train_timesteps - 1 - timesteps) / (scheduler.config.num_train_timesteps - 1)
    continuous_t = continuous_t.to(device=device, dtype=dtype)

    # 获取 alpha 和 sigma
    # x_t = alpha_t * x_0 + sigma_t * noise
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=dtype)
    
    # 使用 torch.gather 从 schedule 中获取对应时间步的值
    alpha_t = torch.gather(alphas_cumprod, 0, timesteps).sqrt()
    sigma_t = (1 - torch.gather(alphas_cumprod, 0, timesteps)).sqrt()

    # 使用数值方法计算导数 (中心差分法)
    # h 是一个微小的时间步长
    h = 1.0 / (scheduler.config.num_train_timesteps - 1)
    
    # t-h 和 t+h 对应的前后时间步
    prev_timesteps = torch.clamp(timesteps - 1, 0, scheduler.config.num_train_timesteps - 1)
    next_timesteps = torch.clamp(timesteps + 1, 0, scheduler.config.num_train_timesteps - 1)
    
    alpha_t_prev = torch.gather(alphas_cumprod, 0, prev_timesteps).sqrt()
    alpha_t_next = torch.gather(alphas_cumprod, 0, next_timesteps).sqrt()

    sigma_t_prev = (1 - torch.gather(alphas_cumprod, 0, prev_timesteps)).sqrt()
    sigma_t_next = (1 - torch.gather(alphas_cumprod, 0, next_timesteps)).sqrt()

    # d(alpha)/dt ≈ (alpha(t+h) - alpha(t-h)) / (2h)
    # 注意：在扩散模型的标准时间表示中，t=0是噪声，t=T是清晰图像，所以导数符号可能需要调整。
    # 这里我们遵循 Diff2Flow 的路径定义，从 x_0 到 x_1。
    # 论文中通常使用从 t=0 (数据) 到 t=1 (噪声) 的连续时间。
    # 如果 scheduler 时间步从大到小，我们的 continuous_t 从小到大，这是匹配的。
    # 导数计算： (f(t_next) - f(t_prev)) / (2*h)，这里 t_next > t_prev。
    alpha_dot_t = (alpha_t_next - alpha_t_prev) / (2 * h)
    sigma_dot_t = (sigma_t_next - sigma_t_prev) / (2 * h)

    # 调整维度以便于广播
    return tuple(map(lambda x: x.view(-1, 1, 1, 1), [alpha_t, sigma_t, alpha_dot_t, sigma_dot_t]))

