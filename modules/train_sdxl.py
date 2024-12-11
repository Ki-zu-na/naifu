import safetensors
import torch
import os
import lightning as pl
from omegaconf import OmegaConf
from diffusers import FlowMatchEulerDiscreteScheduler
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.noise_scheduler, FlowMatchEulerDiscreteScheduler):
            # 确保在初始化时设置时间步
            self.noise_scheduler.set_timesteps(
                self.config.noise_scheduler.params.get("num_train_timesteps", 1000)
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
            latents = batch["pixels"]

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

        # 根据 scheduler 类型选择不同的时间步采样和加噪方式
        if isinstance(self.noise_scheduler, FlowMatchEulerDiscreteScheduler):
            # FlowMatch 训练逻辑
            if not hasattr(self.noise_scheduler, '_timesteps') or len(self.noise_scheduler.timesteps) == 0:
                self.noise_scheduler.set_timesteps(
                    self.config.noise_scheduler.params.get("num_train_timesteps", 1000)
                )
                self.noise_scheduler.timesteps = self.noise_scheduler.timesteps.to(self.target_device)

            # 使用 logit normal 采样
            u = torch.normal(mean=0.0, std=1.0, size=(bsz,), device=self.target_device)
            u = torch.sigmoid(u)

            num_timesteps = len(self.noise_scheduler.timesteps)
            indices = (u * (num_timesteps - 1)).long().clamp(0, num_timesteps - 1)
            timesteps = self.noise_scheduler.timesteps[indices].to(device=latents.device)
            sigmas = self.get_sigmas(timesteps, latents)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
            
            
            # 使用不同的损失计算方式
            model_pred = self.model(noisy_latents.to(model_dtype), timesteps, cond)
            model_pred = model_pred * (-sigmas) + noisy_latents
            weighting = torch.ones_like(sigmas)
            loss = torch.mean(
                (
                    weighting.float() * (model_pred.float() - target.float()) ** 2
                ).reshape(latents.shape[0], -1),
                1,
            )
            loss = loss.mean()
        else:
            # 原有 DDPM 训练逻辑
            timestep_start = advanced.get("timestep_start", 0)
            timestep_end = advanced.get("timestep_end", 1000)
            timestep_sampler_type = advanced.get("timestep_sampler_type", "uniform")

            if timestep_sampler_type == "logit_normal":
                mu = advanced.get("timestep_sampler_mean", 0)
                sigma = advanced.get("timestep_sampler_std", 1)
                t = torch.sigmoid(mu + sigma * torch.randn(size=(bsz,), device=latents.device))
                timesteps = t * (timestep_end - timestep_start) + timestep_start
                timesteps = timesteps.long()
            else:
                timesteps = torch.randint(
                    low=timestep_start,
                    high=timestep_end,
                    size=(bsz,),
                    dtype=torch.int64,
                    device=latents.device,
                )
                timesteps = timesteps.long()

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            noisy_latents = noisy_latents.to(model_dtype)
            noise_pred = self.model(noisy_latents, timesteps, cond)

            is_v = advanced.get("v_parameterization", False)
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps) if is_v else noise

            min_snr_gamma = advanced.get("min_snr", False)
            if min_snr_gamma:
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none")
                loss = loss.mean([1, 2, 3])
                loss = apply_snr_weight(loss, timesteps, self.noise_scheduler, advanced.min_snr_val, is_v)
                loss = loss.mean()
            else:
                loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise FloatingPointError("Error infinite or NaN loss detected")

        return loss

    def get_sigmas(self, timesteps, latents):
        # 确保 scheduler 已经初始化
        if not hasattr(self.noise_scheduler, '_timesteps') or len(self.noise_scheduler.timesteps) == 0:
            self.noise_scheduler.set_timesteps(
                self.config.noise_scheduler.params.get("num_train_timesteps", 1000)
            )
            self.noise_scheduler.timesteps = self.noise_scheduler.timesteps.to(self.target_device)

        sigmas = self.noise_scheduler.sigmas.to(device=self.target_device, dtype=latents.dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.target_device)
        timesteps = timesteps.to(self.target_device)
        step_indices = torch.tensor(
            [min(max(0, (schedule_timesteps == t).nonzero().item() if len((schedule_timesteps == t).nonzero()) > 0 else 0), len(sigmas)-1) 
             for t in timesteps],
            device=self.target_device
        )

        sigma = sigmas[step_indices].flatten()

        while len(sigma.shape) < latents.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma
