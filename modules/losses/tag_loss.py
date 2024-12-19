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
        base_acc = sum(base_loss) / batch_len
        
        if self.total_loss <= 0.0:
            self.total_loss = base_acc
        else:
            batch_beta = min(self.beta, self.global_step / 10.0)
            self.total_loss = (self.total_loss * batch_beta) + (base_acc * (1.0 - batch_beta))
            
        for i in range(batch_len):
            base_mult = 1
            sample_tags = prompts[i].split(self.tag_sep)
            sample_loss = base_loss[i]
            
            tag_mults = []
            base_mults = []
            
            adjust_tags = list(filter(self.check_fn, sample_tags))
            
            for tag in adjust_tags:
                self.tag_counter[tag] = self.tag_counter.get(tag, 0) + 1
                count = self.tag_counter[tag]
                base_mults.append(self.freq_scale[count])
                
                if tag in self.loss_stats:
                    tag_loss, tag_count = self.loss_stats[tag]
                    tag_beta = min(self.beta, tag_count / 10.0)
                    self.loss_stats[tag] = (tag_loss * tag_beta + sample_loss * (1.0 - tag_beta), tag_count + 1)
                else:
                    self.loss_stats[tag] = (sample_loss, 1)
                tag_mults.append(self.loss_stats[tag][0])
                
            for tag in sample_tags:
                if tag in self.tag_rewards:
                    base_mult *= self.tag_rewards[tag]
                    
            if base_mults:
                base_mult *= np.array(base_mults).mean()
                
            hist_loss = np.array(tag_mults).mean() if tag_mults else sample_loss
            target_loss = (sample_loss * (1.0 - self.alpha)) + (hist_loss * self.alpha)
            target_loss *= base_mult
            
            loss_weight = target_loss / base_acc
            loss_weight = 1.0 + self.strength * (loss_weight - 1.0)
            weights.append(loss_weight)
            
        return torch.tensor(weights, device=base_loss.device) 