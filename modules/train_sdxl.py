import safetensors
import torch
import os
import lightning as pl
import torch.nn.functional as F
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger
from modules.sdxl_model import StableDiffusionModel
from modules.scheduler_utils import apply_snr_weight
from lightning.pytorch.utilities.model_summary import ModelSummary
from pathlib import Path
import subprocess
import torch.distributed as dist

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
    model._fabric_wrapped = fabric
    return model, dataset, dataloader, optimizer, scheduler

def get_sigmas(sch, timesteps, n_dim=4, dtype=torch.float32, device="cuda:0"):
    sigmas = sch.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = sch.timesteps.to(device)
    timesteps = timesteps.to(device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

class SupervisedFineTune(StableDiffusionModel):    
    def __init__(self, model_path, config, device):
        super().__init__(model_path, config, device)
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

        cond = self.encode_batch(batch)

        if advanced.get("condition_dropout_rate", 0.0) > 0.0:
            cond = self.dropout_cond(cond)

        model_dtype = next(self.model.parameters()).dtype
        cond = {k: v.to(model_dtype) for k, v in cond.items()}

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents, dtype=model_dtype)
        if advanced.get("offset_noise"):
            offset = torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
            noise = torch.randn_like(latents) + float(advanced.get("offset_noise_val")) * offset

        bsz = latents.shape[0]

        # Sample a random timestep for each image
        timestep_start = advanced.get("timestep_start", 0)
        timestep_end = advanced.get("timestep_end", 1000)
        timestep_sampler_type = advanced.get("timestep_sampler_type", "uniform")

        # Sample a random timestep for each image
        if timestep_sampler_type == "logit_normal":  
            mu = advanced.get("timestep_sampler_mean", 0)
            sigma = advanced.get("timestep_sampler_std", 1)
            t = torch.sigmoid(mu + sigma * torch.randn(size=(bsz,), device=latents.device))
            timesteps = t * (timestep_end - timestep_start) + timestep_start  # scale to [min_timestep, max_timestep)
            timesteps = timesteps.long()
        else:
            # default impl
            timesteps = torch.randint(
                low=timestep_start, 
                high=timestep_end,
                size=(bsz,),
                dtype=torch.int64,
                device=latents.device,
            )
            timesteps = timesteps.long()
 
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(model_dtype)

        # Predict the noise residual and calculate loss
        noise_pred = self.model(noisy_latents, timesteps, cond)

        # Get the target for loss depending on the prediction type
        is_v = advanced.get("v_parameterization", False)
        target = self.noise_scheduler.get_velocity(latents, noise, timesteps) if is_v else noise
        
        base_loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3])

        if hasattr(self, "tag_loss_module"):
            self.tag_loss_module.global_step = self.global_step
            weights = self.tag_loss_module.calculate_loss_weights(batch["prompts"], base_loss.detach())

            # 记录日志
            if hasattr(self, "log_dict"):
                log_dict = {
                    "train/base_loss": base_loss.mean().item(),
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

            loss_weights = weights # tag loss 的权重
        else:
            loss_weights = 1.0 # 默认权重为 1.0

        if advanced.get("min_snr", False):
            # 第一阶段：官方 min-SNR re-weight
            base_loss = apply_snr_weight(
                base_loss, timesteps, self.noise_scheduler,
                advanced.min_snr_val, is_v
            )

            # 第二阶段：自定义权重，按目标类型区分
            snr_t = torch.stack(
                [self.noise_scheduler.all_snr[t] for t in timesteps]
            )
            eps = 1e-8                              # 避免除零
            if is_v:                                # v-prediction
                weight_snr = torch.sqrt(snr_t / (snr_t + 1.0 + eps))
            else:                                   # epsilon-prediction
                weight_snr = 1.0 / torch.sqrt(snr_t + eps)

            weight_snr = torch.clamp(weight_snr, max=10.0)  # 防爆
            loss_weights = loss_weights * weight_snr

        loss = (base_loss * loss_weights).mean()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error(f"Error: NaN or Inf loss encountered! Loss value: {loss.item()}")
            # Raising an error will stop the training process
            raise RuntimeError("NaN or Inf loss detected, stopping training.")
        
        return loss
