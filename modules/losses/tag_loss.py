import torch
from bisect import bisect_left
from collections import UserDict
from typing import Callable, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

DEFAULT_SCALE = {
    -1: 1.1,
    50: 1.05,
    100: 1.02,
    500: 1.01,
    750: 1.00,
    1000: 0.999,
    2000: 0.995,
    4000: 0.99,
    6000: 0.98,
    8000: 0.97,
    10000: 0.96,
    15000: 0.95,
    20000: 0.90,
    30000: 0.85,
    40000: 0.80,
    50000: 0.75,
    100000: 0.70,
}

class TagFreqScale(UserDict):
    def __init__(self, scales=DEFAULT_SCALE):
        super().__init__(data=scales)
        self.data = {int(k): v for k, v in scales.items()}
        self.steps = sorted(self.data.keys())
    
    def __getitem__(self, key):
        key = int(key) if not isinstance(key, int) else key
        
        if key not in self.data:
            idx = bisect_left(self.steps, key)
            if idx >= len(self.steps):
                key = self.steps[-1]
            elif idx == 0:
                key = self.steps[0]
            else:
                key = self.steps[idx]
        return self.data[key]

class TagLossModule:
    def __init__(
        self,
        check_fn: Callable,
        alpha: float = 0.2,
        beta: float = 0.99,
        strength: float = 1.0,
        freq_scale: dict = DEFAULT_SCALE,
        tag_rewards: dict = None,
        tag_sep: str = ", "
    ):
        self.check_fn = check_fn
        self.alpha = alpha
        self.beta = beta
        self.strength = strength
        self.freq_scale = TagFreqScale(freq_scale)
        self.tag_rewards = tag_rewards or {}
        self.tag_sep = tag_sep
        
        self.tag_counter = {}
        self.loss_stats = {}
        self.total_loss = 0.0
        self.global_step = 0
        
    def calculate_loss_weights(self, prompts, base_loss):
        weights = []
        batch_len = len(prompts)
        base_acc = sum(base_loss).item() / batch_len
        
        if self.total_loss <= 0.0:
            self.total_loss = base_acc
        else:
            batch_beta = min(self.beta, self.global_step / 10.0)
            self.total_loss = (self.total_loss * batch_beta) + (base_acc * (1.0 - batch_beta))
            
        for i in range(batch_len):
            base_mult = 1  # 先初始化为1
            sample_tags = prompts[i].split(self.tag_sep)
            sample_loss = base_loss[i].item()

            # 对所有标签应用频率缩放
            for tag in sample_tags:
                self.tag_counter[tag] = self.tag_counter.get(tag, 0) + 1
                count = self.tag_counter[tag]
                base_mult *= self.freq_scale[count]  # 应用频率缩放

            tag_mults = [] # 特殊标签的历史损失
            
            adjust_tags = list(filter(self.check_fn, sample_tags)) # 过滤出特殊标签
            
            for tag in adjust_tags: # 只对特殊标签计算历史损失
                if tag in self.loss_stats:
                    tag_loss, tag_count = self.loss_stats[tag]
                    tag_beta = min(self.beta, tag_count / 10.0)
                    self.loss_stats[tag] = (tag_loss * tag_beta + sample_loss * (1.0 - tag_beta), tag_count + 1)
                else:
                    self.loss_stats[tag] = (sample_loss, 1)
                tag_mults.append(self.loss_stats[tag][0]) # 记录特殊标签的历史损失
                
            for tag in sample_tags:
                if tag in self.tag_rewards:
                    base_mult *= self.tag_rewards[tag] # 应用奖励

            hist_loss = np.mean(tag_mults) if tag_mults else sample_loss # 只使用特殊标签的历史损失
            target_loss = (sample_loss * (1.0 - self.alpha)) + (hist_loss * self.alpha) # 混合损失
            target_loss *= base_mult

            loss_weight = target_loss / base_acc
            loss_weight = 1.0 + self.strength * (loss_weight - 1.0)
            weights.append(loss_weight)

        return torch.tensor(weights, device=base_loss.device, dtype=base_loss.dtype)

    # 添加获取状态的方法
    def get_state_dict(self):
        """返回TagLossModule的状态字典，用于保存"""
        return {
            "tag_counter": self.tag_counter,
            "loss_stats": self.loss_stats,
            "total_loss": self.total_loss,
            "global_step": self.global_step
        }
    
    # 添加加载状态的方法
    def load_state_dict(self, state_dict):
        """从状态字典加载TagLossModule的状态"""
        if not state_dict:
            logger.warning("提供的TagLoss状态字典为空，将使用初始状态")
            return False
            
        self.tag_counter = state_dict.get("tag_counter", {})
        self.loss_stats = state_dict.get("loss_stats", {})
        self.total_loss = state_dict.get("total_loss", 0.0)
        self.global_step = state_dict.get("global_step", 0)
        
        # 记录加载的标签数量
        tag_count = len(self.tag_counter)
        stats_count = len(self.loss_stats)
        logger.info(f"已加载TagLoss状态: {tag_count}个标签计数, {stats_count}个标签统计")
        return True