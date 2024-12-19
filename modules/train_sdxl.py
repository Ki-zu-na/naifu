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
        
        # 初始化tag loss模块
        if config.advanced.get("use_tag_loss", False):
            from modules.losses.tag_loss import TagLossModule
            
            def is_special_tag(tag: str) -> bool:
                return tag.startswith(("artist:", "character:", "style:"))
            
            self.tag_loss_module = TagLossModule(
                check_fn=is_special_tag,
                alpha=config.advanced.get("tag_loss_alpha", 0.2),
                beta=config.advanced.get("tag_loss_beta", 0.99),
                strength=config.advanced.get("tag_loss_strength", 1.0),
                tag_rewards=config.advanced.get("tag_rewards", {})
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
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual and calculate loss
        noisy_latents = noisy_latents.to(model_dtype)
        
        if hasattr(self, "tag_loss_module"):
            # 更新全局步数
            self.tag_loss_module.global_step = self.global_step
            
            if min_snr_gamma:
                base_loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                base_loss = base_loss.mean([1, 2, 3])  # 保持每个样本的损失独立
                
                # 计算权重并应用
                weights = self.tag_loss_module.calculate_loss_weights(
                    batch["prompts"],
                    base_loss.detach()
                )
                
                # 记录详细的日志
                if hasattr(self, "log_dict"):
                    self.log_dict({
                        "train/base_loss": base_loss.mean().item(),
                        "train/base_loss_std": base_loss.std().item(),
                        "train/tag_loss_weight": weights.mean().item(),
                        "train/tag_loss_std": weights.std().item(),
                        "train/weighted_loss": (base_loss * weights).mean().item(),
                        "train/max_weight": weights.max().item(),
                        "train/min_weight": weights.min().item()
                    })
                    
                    # 记录特殊标签的数量
                    special_tags_count = sum(1 for prompt in batch["prompts"] 
                                          for tag in prompt.split(",") 
                                          if self.tag_loss_module.check_fn(tag.strip()))
                    self.log_dict({
                        "train/special_tags_count": special_tags_count
                    })
                
                # 存储metrics供trainer使用
                self.tag_loss_metrics = {
                    "train/base_loss": base_loss.mean().item(),
                    "train/base_loss_std": base_loss.std().item(),
                    "train/tag_loss_weight": weights.mean().item(),
                    "train/tag_loss_std": weights.std().item(),
                    "train/weighted_loss": (base_loss * weights).mean().item(),
                    "train/max_weight": weights.max().item(),
                    "train/min_weight": weights.min().item(),
                    "train/special_tags_count": special_tags_count
                }
                
                # 应用SNR权重和其他权重
                if min_snr_gamma:
                    base_loss = apply_snr_weight(base_loss, timesteps, self.noise_scheduler, advanced.min_snr_val, is_v)
                
                snr_t = torch.stack([self.noise_scheduler.all_snr[t] for t in timesteps])
                snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)
                weight = 1 / torch.sqrt(snr_t)
                
                # 将所有权重相乘并应用到损失
                final_weights = weights * weight
                loss = (base_loss * final_weights).mean()
            else:
                # 对于没有使用min_snr_gamma的情况
                base_loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                base_loss = base_loss.mean([1, 2, 3])
                
                weights = self.tag_loss_module.calculate_loss_weights(
                    batch["prompts"],
                    base_loss.detach()
                )
                loss = (base_loss * weights).mean()
        else:
            noise_pred = self.model(noisy_latents, timesteps, cond)

            # Get the target for loss depending on the prediction type
            is_v = advanced.get("v_parameterization", False)
            if is_v:
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = noise
            
            min_snr_gamma = advanced.get("min_snr", False)            
            if min_snr_gamma:
                # do not mean over batch dimension for snr weight or scale v-pred loss
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])

                if min_snr_gamma:
                    loss = apply_snr_weight(loss, timesteps, self.noise_scheduler, advanced.min_snr_val, is_v)
                    
                # add debiased estimation loss
                # --------------
                snr_t = torch.stack([self.noise_scheduler.all_snr[t] for t in timesteps])  # batch_size
                snr_t = torch.minimum(snr_t, torch.ones_like(snr_t) * 1000)  # if timestep is 0, snr_t is inf, so limit it to 1000
                weight = 1 / torch.sqrt(snr_t)
                loss = weight * loss
                # --------------
                loss = loss.mean()  # mean over batch dimension
            else:
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss
