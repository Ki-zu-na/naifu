import os
import gc
import re
import time

import torch
import lightning as pl

from common.utils import *
from common.logging import logger
from omegaconf import OmegaConf
from pathlib import Path

class Trainer:
    def __init__(self, fabric: pl.Fabric, config: OmegaConf):
        """
        Initialize the trainer with the given fabric and configuration.

        Args:
            fabric (pl.Fabric): The PyTorch Lightning Fabric instance.
            config (OmegaConf): The configuration object.
        """
        self.fabric = fabric
        model_cls = get_class(config.target)
        model, dataset, dataloader, optimizer, scheduler = model_cls(fabric, config)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.dataloader = dataloader
        self.global_step = int(config.get("global_step", 0))
        self.current_epoch = int(config.get("current_epoch", 0))

    def prepare_logger(self):
        """Prepare the logger and log hyperparameters if the logger is not CSVLogger."""
        fabric = self.fabric
        if fabric.logger and fabric.logger.__class__.__name__ != "CSVLogger":
            config = OmegaConf.to_container(self.model.config, resolve=True)
            fabric.logger.log_hyperparams(config)

    def on_post_training_batch(self, is_last=False):
        """
        Perform actions after each training batch.
        """
        if self.fabric.logger and not is_last:
            self.log_lr_values()

        self.perform_sampling(is_last=is_last)
        self.save_model(is_last=is_last)
        self.eval_model(is_last=is_last)

    def log_lr_values(self):
        """
        Log learning rate values for the optimizer.
        """
        optimizer_name = self.model.config.optimizer.name
        last_lr = [group.get("lr", 0) for group in self.optimizer.param_groups]
        ocls = self.optimizer.__class__.__name__

        for i, lr in enumerate(last_lr):
            self.fabric.log(f"lr/{ocls}-{i}", lr, step=self.global_step)

        is_da = optimizer_name.startswith("DAdapt")
        is_prodigy = optimizer_name.startswith("prodigyopt")
        if not (is_da or is_prodigy):
            return

        last_d_lr = [(g["d"] * g["lr"]) for g in self.optimizer.param_groups]
        for i, lr in enumerate(last_d_lr):
            self.fabric.log(f"d*lr/{ocls}-{i}", lr, step=self.global_step)

    def eval_model(self, is_last: bool = False):
        """
        Save the model checkpoint.
        """
        config = self.model.config
        cfg = config.trainer
        eval_st = cfg.get("eval_steps", -1)
        eval_fq = cfg.get("eval_epochs", -1)

        is_eval_step = eval_st > 0 and self.global_step % eval_st == 0
        is_eval_epoch = eval_fq > 0 and self.current_epoch % eval_fq == 0
        
        should_eval = (is_last and is_eval_epoch) or is_eval_step
        has_eval_method = hasattr(self.model, "eval_model")
        
        if not should_eval or not has_eval_method:
            return

        if "schedulefree" in self.optimizer.__class__.__name__.lower():
            self.optimizer.eval()
            
        self.model.eval_model(
            logger=self.fabric.logger,
            current_epoch=self.current_epoch,
            global_step=self.global_step,
        )
        torch.cuda.empty_cache()
        
        if "schedulefree" in self.optimizer.__class__.__name__.lower():
            self.optimizer.train()

    def save_model(self, is_last: bool = False):
        """
        Save the model checkpoint.
        """
        config = self.model.config
        cfg = config.trainer
        ckpt_st = cfg.checkpoint_steps
        ckpt_fq = cfg.checkpoint_freq
        ckpt_dir = cfg.checkpoint_dir
        max_ckpts = cfg.get("max_checkpoints", -1)

        is_ckpt_step = ckpt_st > 0 and self.global_step % ckpt_st == 0
        is_ckpt_epoch = ckpt_fq > 0 and self.current_epoch % ckpt_fq == 0
        should_save = (is_last and is_ckpt_epoch) or is_ckpt_step
        if not should_save:
            return
        
        if "schedulefree" in self.optimizer.__class__.__name__.lower():
            self.optimizer.eval()

        postfix = f"e{self.current_epoch}_s{self.global_step}"
        model_path = os.path.join(ckpt_dir, f"checkpoint-{postfix}")
        save_weights_only = cfg.get("save_weights_only", False)
        
        logger.info("Saving model checkpoint")
        metadata = {
            "global_step": str(self.global_step),
            "current_epoch": str(self.current_epoch),
        }
        
        if max_ckpts > 0:
            existing_ckpts = []
            for f in os.listdir(ckpt_dir):
                if f.startswith("checkpoint-") and f.endswith(".ckpt"):
                    ckpt_path = os.path.join(ckpt_dir, f)
                    state_path = ckpt_path.replace(".ckpt", "_state.pt")
                    existing_ckpts.append((ckpt_path, os.path.getmtime(ckpt_path)))
            
            existing_ckpts.sort(key=lambda x: x[1])
            
            while len(existing_ckpts) >= max_ckpts:
                oldest_ckpt, _ = existing_ckpts.pop(0)
                if os.path.exists(oldest_ckpt):
                    os.remove(oldest_ckpt)
                    logger.info(f"Removed old checkpoint: {oldest_ckpt}")
                
                state_path = oldest_ckpt.replace(".ckpt", "_state.pt")
                if os.path.exists(state_path):
                    os.remove(state_path)
                    logger.info(f"Removed old state file: {state_path}")
        
        if not save_weights_only:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,

            }
            self.fabric.save(model_path + "_state.pt", state)
            logger.info(f"Saving model state to {model_path}_state.pt")
        
        self.model.save_checkpoint(model_path, metadata)
        
        if "schedulefree" in self.optimizer.__class__.__name__.lower():
            self.optimizer.train()

    def perform_sampling(self, is_last: bool = False):
        """
        Perform image/text sampling.
        """
        config = self.model.config
        enabled_sampling = config.sampling.enabled and hasattr(self.model, "generate_samples")

        sampling_cfg = config.sampling
        sampling_steps = sampling_cfg.every_n_steps
        sample_by_step = sampling_steps > 0 and self.global_step % sampling_steps == 0
        sampling_epochs = sampling_cfg.every_n_epochs
        sample_by_epoch = sampling_epochs > 0 and self.current_epoch % sampling_epochs == 0
        sample_on_start = config.sampling.get("sample_on_start", False) \
            and not getattr(self, "sampler_initialized", False)

        if not enabled_sampling or len(sampling_cfg.prompts) == 0:
            return
        
        if sampling_cfg.get("save_dir", None):
            os.makedirs(sampling_cfg.save_dir, exist_ok=True)

        if (is_last and sample_by_epoch) or sample_by_step or sample_on_start:
            setattr(self, "sampler_initialized", True)
            
            if "schedulefree" in self.optimizer.__class__.__name__.lower():
                self.optimizer.eval()
    
            torch.cuda.empty_cache()
            rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state()

            self.model.generate_samples(
                logger=self.fabric.logger,
                current_epoch=self.current_epoch,
                global_step=self.global_step,
            )
            torch.cuda.empty_cache()
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_rng_state)
            
            if "schedulefree" in self.optimizer.__class__.__name__.lower():
                self.optimizer.train()
            
    def train_loop(self):
        """
        Run the training loop.
        """
        config = self.model.config
        fabric = self.fabric
        cfg = config.trainer
        target_device = fabric.device
        
        # # 在训练开始时将模型移至目标设备
        # def move_to_device(model):
        #     if hasattr(model, 'to'):
        #         return model.to(target_device)
        #     return model

        # # 移动主要模型组件
        # self.model = move_to_device(self.model)
        # if hasattr(self.model, 'first_stage_model'):
        #     self.model.first_stage_model = move_to_device(self.model.first_stage_model)
        # if hasattr(self.model, 'model'):
        #     self.model.model = move_to_device(self.model.model)
        # if hasattr(self.model, 'unet_ref'):
        #     self.model.unet_ref = move_to_device(self.model.unet_ref)
        
        # # 移动 LyCORIS 相关组件
        # if hasattr(self.model, 'lycoris_unet'):
        #     self.model.lycoris_unet = move_to_device(self.model.lycoris_unet)
        # if hasattr(self.model, 'lycoris_te1'):
        #     self.model.lycoris_te1 = move_to_device(self.model.lycoris_te1)
        # if hasattr(self.model, 'lycoris_te2'):
        #     self.model.lycoris_te2 = move_to_device(self.model.lycoris_te2)
        
        # # 移动文本编码器
        # if hasattr(self.model, 'text_encoder_1'):
        #     self.model.text_encoder_1 = move_to_device(self.model.text_encoder_1)
        # if hasattr(self.model, 'text_encoder_2'):
        #     self.model.text_encoder_2 = move_to_device(self.model.text_encoder_2)

        grad_accum_steps = cfg.accumulate_grad_batches
        grad_clip_val = cfg.gradient_clip_val

        local_step = 0
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        latest_ckpt = get_latest_checkpoint(cfg.checkpoint_dir)
        
        def log_device_info():
            if fabric.is_global_zero:
                logger.info(f"Training on device: {target_device}")
                if hasattr(self.model, 'model'):
                    logger.info(f"Main model device: {next(self.model.model.parameters()).device}")
                if hasattr(self.model, 'unet_ref'):
                    logger.info(f"UNet ref device: {next(self.model.unet_ref.parameters()).device}")
                if hasattr(self.model, 'lycoris_unet'):
                    logger.info(f"LyCORIS UNet device: {next(self.model.lycoris_unet.parameters()).device}")

        log_device_info()
        # # load state if you used fabric.save
        # state = dict(
        #     mode=self.model,
        #     global_step=self.global_step, 
        #     current_epoch=self.current_epoch,
        #     optimizer=self.optimizer,
        # )
        # self.fabric.load(model_path + "_state.pt", state)

        if cfg.get("resume"):
            if latest_ckpt:
                state_path = latest_ckpt.replace(".ckpt", "_state.pt")
                if os.path.exists(state_path):
                    logger.info(f"从完整状态文件恢复训练: {state_path}")
                    state = self.fabric.load(state_path)
                    
                    # 恢复模型状态
                    self.model.load_state_dict(state["model"])
                    # 恢复优化器状态
                    self.optimizer.load_state_dict(state["optimizer"])
                    # 恢复调度器状态
                    if self.scheduler and state["scheduler"]:
                        self.scheduler.load_state_dict(state["scheduler"])
                    
                    # 恢复训练步数和轮数
                    self.global_step = state["global_step"]
                    self.current_epoch = state["current_epoch"]
                    
                    
                    logger.info(f"成功恢复训练状态到步数 {self.global_step} 和轮数 {self.current_epoch}")
                else:
                    # 修改正则表达式以匹配两种文件格式
                    match = re.search(r'checkpoint-e(\d+)_s(\d+)(?:_state)?(?:\.ckpt|\.pt)', os.path.basename(latest_ckpt))
                    if match:
                        self.current_epoch = int(match.group(1))
                        self.global_step = int(match.group(2))
                        logger.info(f"从文件名提取训练状态：轮数 {self.current_epoch}，步数 {self.global_step}")
                    
                    # 然后尝试从优化器状态或checkpoint中获取信息
                    opt_name = Path(latest_ckpt).stem + "_optimizer"
                    opt_path = Path(latest_ckpt).with_stem(opt_name).with_suffix(".pt")
                    if opt_path.is_file():
                        remainder = fabric.load(opt_path, {"optimizer": self.optimizer})
                        logger.info(f"Loaded optimizer state from {opt_path}")
                        self.global_step = int(remainder.pop("global_step", self.global_step))
                        self.current_epoch = int(remainder.pop("current_epoch", self.current_epoch))
                    else:
                        if latest_ckpt.endswith(".ckpt"):
                            sd = torch.load(latest_ckpt, map_location="cpu")
                            self.global_step = int(sd.pop("global_step", self.global_step))
                            self.current_epoch = int(sd.pop("current_epoch", self.current_epoch))
                        elif latest_ckpt.endswith(".safetensors"):
                            with safetensors.torch.safe_open(latest_ckpt, framework="pt") as f:
                                metadata = f.metadata()
                                self.global_step = int(metadata.get("global_step", self.global_step))
                                self.current_epoch = int(metadata.get("current_epoch", self.current_epoch))
                
                    logger.info(f"Resuming training from step {self.global_step} and epoch {self.current_epoch}")

                    if 'sd' in locals():
                        del sd
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.memory_summary(device=None, abbreviated=False)
            else:
                # 当没有找到 latest_ckpt 时，尝试从 model_path 中提取信息
                model_path = cfg.get("model_path", "")
                if model_path:
                    match = re.search(r'checkpoint-e(\d+)_s(\d+)\.ckpt', os.path.basename(model_path))
                    if match:
                        self.current_epoch = int(match.group(1))
                        self.global_step = int(match.group(2))
                        logger.info(f"No latest checkpoint found. Extracted epoch {self.current_epoch} and step {self.global_step} from model path.")
                    else:
                        logger.warn("No latest checkpoint found and couldn't extract information from model path.")
                else:
                    logger.info("No latest checkpoint found and no model path provided.")
        else:
            logger.info(f"Starting training from epoch {self.current_epoch} and step {self.global_step}")

        # 添加一个批次处理函数来确保数据在正确的设备上
        # def process_batch(batch):
        #     if isinstance(batch, torch.Tensor):
        #         return batch.to(target_device)
        #     elif isinstance(batch, dict):
        #         return {k: process_batch(v) for k, v in batch.items()}
        #     elif isinstance(batch, (list, tuple)):
        #         return type(batch)(process_batch(x) for x in batch)
        #     return batch

        should_stop = False
        if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
            should_stop = True

        self.prepare_logger()
        loss_rec = LossRecorder()
        progress  = ProgressBar(
            total=len(self.dataloader) // config.trainer.accumulate_grad_batches,
            disable=not fabric.is_global_zero,
        )
        assert len(self.dataloader) > 0, "Dataloader is empty"
        
        steps_per_epoch = len(self.dataloader) // config.trainer.accumulate_grad_batches
        resume_epoch = self.global_step // steps_per_epoch
        resume_step = self.global_step % steps_per_epoch
        while not should_stop:
            # This is the beginning of each epoch
            loss_rec.reset()  
            desc = f"Epoch {self.current_epoch}"
            progress.update(desc, 0)
            torch.cuda.empty_cache()
            
            if "schedulefree" in self.optimizer.__class__.__name__.lower():
                self.optimizer.train()

            for batch_idx, batch in enumerate(self.dataloader):  
                # batch = process_batch(batch)
                # Skip the completed steps in the current epoch
                local_acc_step = batch_idx // grad_accum_steps + 1
                if self.current_epoch == resume_epoch and local_acc_step < resume_step:
                    continue
                
                local_step += 1    
                local_timer = time.perf_counter()
                
                is_accumulating = local_step % grad_accum_steps != 0
                fabric_module = getattr(self.model, "model", None)
                if hasattr(self.model, "get_module"):
                    fabric_module = self.model.get_module()
                    
                with fabric.no_backward_sync(fabric_module, enabled=is_accumulating):
                # with torch.autograd.detect_anomaly():
                    loss = self.model(batch)
                    self.fabric.backward(loss / grad_accum_steps)

                loss = loss.detach().item()
                loss_rec.add(epoch=self.current_epoch, step=batch_idx, loss=loss)
                metrics = {
                    "train/loss": loss,
                    "train/avg_loss": loss_rec.avg,
                    "trainer/step_t": time.perf_counter() - local_timer,
                }
                
                # 添加tag loss相关指标到metrics
                if hasattr(self.model, "tag_loss_module"):
                    tag_metrics = getattr(self.model, "tag_loss_metrics", {})
                    metrics.update(tag_metrics)
                    
                epoch_metrics = {
                    "train/epoch_avg_loss": loss_rec.avg,
                    "epoch": self.current_epoch
                }
                stat_str = f"train_loss: {loss:.3f}, avg_loss: {loss_rec.avg:.3f}"
                progress.update(desc, local_acc_step, status=stat_str)
                    
                # skip here if we are accumulating
                if is_accumulating:
                    continue

                if grad_clip_val > 0:
                    grad_norm = self.fabric.clip_gradients(
                        module=fabric_module, 
                        optimizer=self.optimizer, 
                        max_norm=grad_clip_val
                    )
                    if grad_norm is not None:
                        metrics["train/grad_norm"] = grad_norm

                if self.optimizer is not None:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    if "transformers" in config.scheduler.name:
                        self.scheduler.step(self.global_step)
                    else:
                        self.scheduler.step()

                if fabric.logger:
                    fabric.log_dict(metrics=metrics, step=self.global_step)
                    fabric.log_dict(metrics=epoch_metrics, step=self.global_step)

                self.global_step += 1
                self.on_post_training_batch()

            self.current_epoch += 1
            if cfg.max_epochs > 0 and self.current_epoch >= cfg.max_epochs:
                should_stop = True

            self.on_post_training_batch(is_last=True)


        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.memory_summary(device=None, abbreviated=False)    