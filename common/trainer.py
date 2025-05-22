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
            # 检查模型是否有tag_loss_module属性
            tag_loss_state = None
            if hasattr(self.model, "tag_loss_module"):
                tag_loss_state = self.model.tag_loss_module.get_state_dict()
                logger.info(f"保存TagLoss状态: {len(tag_loss_state['tag_counter'])}个标签计数")
            
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "tag_loss_state": tag_loss_state,  # 添加TagLoss状态
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
            resumed_via_full_state = False

            if latest_ckpt:
                # Try to load from a comprehensive state file first
                derived_state_file_path = ""
                if latest_ckpt.endswith(".ckpt"):
                    derived_state_file_path = latest_ckpt.replace(".ckpt", "_state.pt")
                elif latest_ckpt.endswith(".safetensors"):
                    derived_state_file_path = latest_ckpt.replace(".safetensors", "_state.pt")
                # Consider if latest_ckpt could already be a _state.pt file.
                # For this logic, we assume get_latest_checkpoint returns the primary model file.

                if derived_state_file_path and os.path.exists(derived_state_file_path):
                    logger.info(f"Attempting to load full training state from: {derived_state_file_path}")
                    try:
                        objects_to_load = {"model": self.model}
                        if self.optimizer:
                            objects_to_load["optimizer"] = self.optimizer
                        if self.scheduler:
                            objects_to_load["scheduler"] = self.scheduler

                        loaded_extras = self.fabric.load(derived_state_file_path, state=objects_to_load)

                        self.global_step = int(loaded_extras.pop("global_step", self.global_step))
                        self.current_epoch = int(loaded_extras.pop("current_epoch", self.current_epoch))
                        
                        tag_loss_state_from_full = loaded_extras.get("tag_loss_state")
                        if tag_loss_state_from_full and hasattr(self.model, "tag_loss_module"):
                            self.model.tag_loss_module.load_state_dict(tag_loss_state_from_full)
                            logger.info(f"Loaded TagLoss state from full state file: {derived_state_file_path}.")
                        elif hasattr(self.model, "tag_loss_module") and not tag_loss_state_from_full:
                             logger.info(f"TagLoss state not found in {derived_state_file_path} or model has no tag_loss_module for full state load.")
                        
                        logger.info(f"Successfully resumed training from full state file: {derived_state_file_path}. Epoch: {self.current_epoch}, Step: {self.global_step}")
                        resumed_via_full_state = True
                    except Exception as e:
                        logger.warning(f"Failed to load full state from {derived_state_file_path}: {e}. Falling back to older resume methods.")
            
            if not resumed_via_full_state:
                if latest_ckpt:
                    match = re.search(r'checkpoint-e(\d+)_s(\d+)(?:_state)?(?:\.ckpt|\.pt|\.safetensors)', os.path.basename(latest_ckpt))
                    if match:
                        self.current_epoch = int(match.group(1))
                        self.global_step = int(match.group(2))
                        logger.info(f"从文件名提取训练状态：轮数 {self.current_epoch}，步数 {self.global_step}")
                    
                    opt_name = Path(latest_ckpt).stem + "_optimizer"
                    opt_path = Path(latest_ckpt).with_stem(opt_name).with_suffix(".pt")
                    if opt_path.is_file():
                        remainder = self.fabric.load(opt_path, {"optimizer": self.optimizer})
                        logger.info(f"Loaded optimizer state from {opt_path}")
                        self.global_step = int(remainder.pop("global_step", self.global_step))
                        self.current_epoch = int(remainder.pop("current_epoch", self.current_epoch))
                    else:
                        if latest_ckpt.endswith(".ckpt"):
                            sd = torch.load(latest_ckpt, map_location="cpu")
                            self.global_step = int(sd.pop("global_step", self.global_step))
                            self.current_epoch = int(sd.pop("current_epoch", self.current_epoch))
                            # 'sd' will be deleted in the common cleanup section if it exists
                        elif latest_ckpt.endswith(".safetensors"):
                            # Ensure 'import safetensors.torch' is present at the top of the file
                            with safetensors.torch.safe_open(latest_ckpt, framework="pt") as f: # type: ignore
                                metadata = f.metadata()
                                if metadata: # Check if metadata is not None
                                    self.global_step = int(metadata.get("global_step", self.global_step))
                                    self.current_epoch = int(metadata.get("current_epoch", self.current_epoch))
                
                    logger.info(f"Resuming training from step {self.global_step} and epoch {self.current_epoch}")

                    # Attempt to load TagLoss state separately if not loaded via full state
                    # This uses derived_state_file_path if available and file exists
                    if derived_state_file_path and os.path.exists(derived_state_file_path) and hasattr(self.model, "tag_loss_module"):
                        tag_loss_loaded_separately = False
                        logger.info(f"Attempting to separately load TagLoss state from {derived_state_file_path} as full state load was not successful or did not include it.")
                        try:
                            state_dict_for_tag_loss = torch.load(derived_state_file_path, map_location="cpu")
                            tag_loss_state = state_dict_for_tag_loss.get("tag_loss_state", None)
                            
                            if tag_loss_state:
                                self.model.tag_loss_module.load_state_dict(tag_loss_state)
                                tag_loss_loaded_separately = True
                                logger.info(f"从 {derived_state_file_path} 单独加载TagLoss状态成功。")
                        except Exception as e:
                            logger.warning(f"单独加载TagLoss状态时出错: {e}")
                        
                        if not tag_loss_loaded_separately:
                             logger.warning(f"未能从 {derived_state_file_path} 单独加载TagLoss状态，或文件中不存在该状态。")
                    elif hasattr(self.model, "tag_loss_module"):
                        logger.warning("未能加载TagLoss状态（因相应state文件未找到或未定义），将使用初始状态。")

                else: # No latest_ckpt found
                    model_path = cfg.get("model_path", "")
                    if model_path:
                        match = re.search(r'checkpoint-e(\d+)_s(\d+)(?:_state)?(?:\.ckpt|\.pt|\.safetensors)', os.path.basename(model_path))
                        if match:
                            self.current_epoch = int(match.group(1))
                            self.global_step = int(match.group(2))
                            logger.info(f"No latest checkpoint found. Extracted epoch {self.current_epoch} and step {self.global_step} from model path.")
                        else:
                            logger.warn("No latest checkpoint found and couldn't extract information from model path.")
                    else:
                        logger.info("No latest checkpoint found and no model path provided.")

            # Common cleanup after all resume attempts
            # Run if resume was attempted (either via latest_ckpt or model_path in fallback)
            if latest_ckpt or (not resumed_via_full_state and cfg.get("model_path", "")):
                if 'sd' in locals() and not resumed_via_full_state : # sd is only relevant for the fallback ckpt loading path
                    del sd
                torch.cuda.empty_cache()
                gc.collect()
                # torch.cuda.memory_summary(device=None, abbreviated=False) # Optional
        else:
            logger.info(f"Starting training from epoch {self.current_epoch} and step {self.global_step}")



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
                    
                    # 检查loss是否为NaN
                    if torch.isnan(loss):
                        logger.warning(f"检测到NaN loss，跳过当前batch (epoch: {self.current_epoch}, step: {batch_idx})")
                        continue
                        
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