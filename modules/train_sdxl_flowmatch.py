import safetensors
import torch
import os, math
import lightning as pl
from typing import Callable
from omegaconf import OmegaConf
from common.utils import get_class, get_latest_checkpoint, load_torch_file
from common.logging import logger

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

        cond = self.encode_batch(batch)
        model_dtype = next(self.model.parameters()).dtype
        cond = {k: v.to(model_dtype) for k, v in cond.items()}

        bsz = latents.shape[0]
        noise = torch.randn_like(latents, device=latents.device)

        u = torch.normal(mean=0.0, std=1.0, size=(bsz,), device=self.target_device)
        t = torch.nn.functional.sigmoid(u)
        shift = advanced.get("flow_shift", 2.0)

        if advanced.get("use_dynamic_shifting", False):
            latent_h, latent_w = latents.shape[2], latents.shape[3]
            current_seq_len = latent_h * latent_w //16

            base_mu = advanced.get("base_shift", 0.5)
            max_mu = advanced.get("max_shift", 1.15)
            base_len = advanced.get("base_image_seq_len", 256)
            max_len = advanced.get("max_image_seq_len", 4096)
            mu = get_lin_function(x1=base_len, y1=base_mu, x2=max_len, y2=max_mu)(current_seq_len)
            sigmas = time_shift(mu, 1.0, t)
        else:
            sigmas = (t * shift) / (1 + (shift - 1) * t)

        timesteps = (sigmas * 1000.0)
        sigmas = sigmas.view(-1, 1, 1, 1)
        noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
        model_pred = self.model(noisy_latents.to(torch.bfloat16), timesteps, cond)
        target_v = noise - latents
        base_loss = torch.mean(((model_pred.float() - target_v.float()) ** 2).reshape(latents.shape[0], -1), 1)
        
        # --- Start of Min-SNR implementation ---
        # Get gamma from config, e.g., 5.0
        gamma = advanced.get("min_snr_val", 5.0) 
        
        # Calculate SNR from sigmas, add a small epsilon to avoid division by zero
        # sigmas here is a 1D tensor of size [bsz] before being reshaped
        snr = ((1.0 - sigmas)**2) / (sigmas**2 + 1e-9)
        
        # Calculate Min-SNR weights
        min_snr = torch.minimum(snr, torch.full_like(snr, gamma))
        
        # For v-prediction, the paper suggests weighting by min_snr / (snr + 1)
        # Since we are predicting `noise - latents`, which is related to `v`, we use this weighting.
        if advanced.get("v_prediction_like_loss", True):
             snr_weights = min_snr / (snr + 1)
        else:
             snr_weights = min_snr / snr
        
        # --- End of Min-SNR implementation ---

        if hasattr(self, "tag_loss_module"):
            self.tag_loss_module.global_step = self.global_step
            weights = self.tag_loss_module.calculate_loss_weights(
                batch["prompts"],
                base_loss.detach()
            )

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

            # Combine snr_weights with tag_loss weights
            final_weights = weights * snr_weights.detach()
            loss = (base_loss * final_weights).mean()
        else:
            loss = (base_loss * snr_weights).mean()

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