import functools
import math
import os
import numpy as np
import random
import torch
from collections import defaultdict

import cv2
from pathlib import Path
from torch.utils.data import Dataset, get_worker_info
from data.image_storage import DirectoryImageStore, Entry, LatentStore
from torchvision.transforms import Resize, InterpolationMode
from common.logging import logger
from common.utils import get_class

image_suffix = set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"])


def is_latent_folder(path: Path):
    # iterate over all files in the folder and find if any of them is a latent
    for p in path.iterdir():
        if p.is_dir():
            continue
        if p.suffix == ".h5":
            return True

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset: RatioDataset = worker_info.dataset  # type: ignore
    # random.seed(worker_info.seed)  # type: ignore
    dataset.init_batches()


class RatioDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        img_path: Path | str | list,
        ucg: int = 0,
        rank: int = 0,
        dtype=torch.float16,
        seed: int = 42,
        use_central_crop=True,
        **kwargs,
    ):
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.num_workers = kwargs.get("num_workers", 4)
        self.use_central_crop = use_central_crop

        root_path = Path(img_path)
        assert root_path.exists(), f"Path {root_path} does not exist."

        if kwargs.get("store_cls"):
            store_class = get_class(kwargs["store_cls"])
        elif is_latent_folder(root_path):
            store_class = LatentStore
        else:
            store_class = DirectoryImageStore

        self.store = store_class(
            root_path,
            rank=rank,
            ucg=ucg,
            dtype=dtype,
            **kwargs,
        )

    def generate_buckets(self):
        raise NotImplementedError

    def assign_buckets(self):
        img_res = np.array(self.store.raw_res)
        self.to_size = {}
        self.bucket_content = defaultdict(list)

        # 为每个图像分配桶
        for idx, (img_width, img_height) in enumerate(img_res):
            img_area = img_width * img_height

            # 检查图像是否需要调整大小（仅允许缩小）
            target_width, target_height = img_width, img_height  # 初始化为原始尺寸
            if img_area > self.target_area:
                scale_factor = math.sqrt(self.target_area / img_area)
                target_width = math.floor(img_width * scale_factor)
                target_height = math.floor(img_height * scale_factor)

            # 计算目标尺寸（确保是 divisible 的倍数）
            bucket_width = math.floor(target_width / self.divisible) * self.divisible
            bucket_height = math.floor(target_height / self.divisible) * self.divisible

            # 确保最小尺寸
            bucket_width = max(self.divisible, bucket_width)
            bucket_height = max(self.divisible, bucket_height)

            reso = (bucket_height, bucket_width) # 存储为 (height, width)
            self.bucket_content[reso].append(idx)
            self.to_size[idx] = reso

        # 移除太小的桶
        min_bucket_size = self.batch_size // 2
        valid_buckets = {k: v for k, v in self.bucket_content.items()
                        if len(v) >= min_bucket_size}

        # 将小桶中的图像重新分配到最接近的大桶中
        for k, v in self.bucket_content.items():
            if k not in valid_buckets:
                for idx in v:
                    closest_bucket = min(valid_buckets.keys(),
                                      key=lambda x: abs(x[0]/x[1] - k[0]/k[1]))
                    valid_buckets[closest_bucket].append(idx)
                    self.to_size[idx] = closest_bucket

        self.bucket_content = list(valid_buckets.values())


    def init_batches(self):
        self.assign_buckets()
        self.assign_batches()

    def init_dataloader(self, **kwargs):
        dataloader = torch.utils.data.DataLoader(
            self,
            sampler=None,
            batch_size=None,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=True,
            pin_memory=True,
            **kwargs,
        )
        return dataloader

    def __len__(self):
        return len(self.batch_idxs)

    @staticmethod
    @functools.cache
    def fit_dimensions(target_ratio, min_h, min_w):
        min_area = min_h * min_w
        h = max(min_h, math.ceil(math.sqrt(min_area * target_ratio)))
        w = max(min_w, math.ceil(h / target_ratio))

        if w < min_w:
            w = min_w
            h = max(min_h, math.ceil(w * target_ratio))

        while h * w < min_area:
            increment = 8
            if target_ratio >= 1:
                h += increment
            else:
                w += increment

            w = max(min_w, math.ceil(h / target_ratio))
            h = max(min_h, math.ceil(w * target_ratio))
        return int(h), int(w)

    def assign_batches(self):
        self.batch_idxs = []
        for bucket in self.bucket_content:
            if not bucket or len(bucket) == 0:
                continue
            reminder = len(bucket) % self.batch_size
            bucket = np.array(bucket)
            self.rng.shuffle(bucket)
            if not reminder:
                self.batch_idxs.extend(bucket.reshape(-1, self.batch_size))
            else:
                self.batch_idxs.extend(bucket[:-reminder].reshape(-1, self.batch_size))
                self.batch_idxs.append(bucket[-reminder:])

        self.rng.shuffle(self.batch_idxs)

    def __getitem__(self, idx):
        img_idxs = self.batch_idxs[idx]
        return self.store.get_batch(img_idxs)


class AspectRatioDataset(RatioDataset):
    """Original implementation of AspectRatioDataset, equal to other frameworks"""
    def __init__(
        self,
        batch_size: int,
        img_path: Path | str | list,
        ucg: int = 0, rank: int = 0,
        dtype=torch.float16,
        target_area: int = 1024 * 1024,
        min_size: int = 512,
        max_size: int = 2048,
        divisible: int = 64,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(batch_size, img_path, ucg, rank, dtype, seed, **kwargs)
        self.target_area = target_area
        self.max_size, self.min_size, self.divisible = max_size, min_size, divisible
        self.store.crop = self.crop

        self.generate_buckets()
        self.init_batches()

    def crop(self, entry: Entry, i: int) -> Entry:
        assert self.to_size is not None, "to_size is not initialized"
        H, W = entry.pixel.shape[-2:]
        logger.debug(f"Crop function input shape: {entry.pixel.shape}, original size: {(H, W)}")
        target_h, target_w = self.to_size[i] # 直接从 to_size 获取目标尺寸
        logger.debug(f"Target size from to_size: {(target_h, target_w)}")

        if not entry.is_latent:
            resize_h, resize_w = target_h, target_w # resize 目标尺寸就是 bucket 尺寸
            logger.debug(f"Resize target size: {(resize_h, resize_w)}")
            pixel = entry.pixel
            if isinstance(pixel, torch.Tensor):
                pixel = pixel.permute(1, 2, 0).cpu().numpy()

            interp = cv2.INTER_AREA if resize_h < H else cv2.INTER_LANCZOS4
            pixel = cv2.resize(pixel.astype(float), (resize_w, resize_h), interpolation=interp)
            entry.pixel = torch.from_numpy(pixel).permute(2, 0, 1)
        else:
            target_h = target_h // 8
            target_w = target_w // 8
            logger.debug(f"Latent crop size: {(target_h, target_w)}")

        H, W = entry.pixel.shape[-2:]
        if self.use_central_crop:
            dh, dw = (H - target_h) // 2, (W - target_w) // 2 # crop 尺寸也使用 bucket 尺寸
        else:
            assert H >= target_h and W >= target_w, f"{H}<{target_h} or {W}<{target_w}"
            dh, dw = random.randint(0, H - target_h), random.randint(0, W - target_w)

        entry.pixel = entry.pixel[:, dh : dh + target_h, dw : dw + target_w] # crop 尺寸也使用 bucket 尺寸
        logger.debug(f"Cropped to shape: {entry.pixel.shape}, target shape: {(target_h, target_w)}")
        return entry, dh, dw


    def generate_buckets(self):
        assert (
            self.target_area % 4096 == 0
        ), "target area (h * w) must be divisible by 64"
        width = np.arange(self.min_size, self.max_size + 1, self.divisible)
        height = np.minimum(
            self.max_size,
            ((self.target_area // width) // self.divisible) * self.divisible,
        )
        valid_mask = height >= self.min_size

        resos = set(zip(width[valid_mask], height[valid_mask]))
        resos.update(zip(height[valid_mask], width[valid_mask]))
        resos.add(((int(np.sqrt(self.target_area)) // self.divisible) * self.divisible,) * 2)
        self.buckets_sizes = np.array(sorted(resos))
        self.bucket_ratios = self.buckets_sizes[:, 0] / self.buckets_sizes[:, 1]
        self.ratio_to_bucket = {ratio: hw for ratio, hw in zip(self.bucket_ratios, self.buckets_sizes)}

    def assign_buckets(self):
        img_res = np.array(self.store.raw_res)
        img_ratios = img_res[:, 0] / img_res[:, 1]
        self.bucket_content = [[] for _ in range(len(self.buckets_sizes))]
        self.to_ratio = {}

        # Assign images to buckets
        for idx, img_ratio in enumerate(img_ratios):
            diff = np.abs(np.log(self.bucket_ratios) - np.log(img_ratio))
            bucket_idx = np.argmin(diff)
            self.bucket_content[bucket_idx].append(idx)
            self.to_ratio[idx] = self.bucket_ratios[bucket_idx]


class AdaptiveSizeDataset(RatioDataset):
    """AdaptiveRatioDataset, a modified version of AspectRatioDataset which avoid resize from smaller images"""
    def __init__(
        self,
        batch_size: int,
        img_path: Path | str | list,
        ucg: int = 0,
        rank: int = 0,
        dtype=torch.float16,
        target_area: int = 1024 * 1024,
        divisible: int = 64,
        seed: int = 42,
        **kwargs
    ):

        super().__init__(batch_size, img_path, ucg, rank, dtype, seed, **kwargs)
        self.metadata = None # 移除此处 metadata 加载和过滤逻辑

        self.store.crop = self.crop
        self.target_area = target_area
        self.divisible = divisible

        self.generate_buckets()
        self.init_batches()

    def crop(self, entry: Entry, i: int) -> Entry:
        assert self.to_size is not None, "to_size is not initialized"
        H, W = entry.pixel.shape[-2:]
        logger.debug(f"Crop function input shape: {entry.pixel.shape}, original size: {(H, W)}")
        target_h, target_w = self.to_size[i] # 直接从 to_size 获取目标尺寸
        logger.debug(f"Target size from to_size: {(target_h, target_w)}")

        if not entry.is_latent:
            resize_h, resize_w = target_h, target_w # resize 目标尺寸就是 bucket 尺寸
            logger.debug(f"Resize target size: {(resize_h, resize_w)}")
            pixel = entry.pixel
            if isinstance(pixel, torch.Tensor):
                pixel = pixel.permute(1, 2, 0).cpu().numpy()

            interp = cv2.INTER_AREA if resize_h < H else cv2.INTER_LANCZOS4
            pixel = cv2.resize(pixel.astype(float), (resize_w, resize_h), interpolation=interp)
            entry.pixel = torch.from_numpy(pixel).permute(2, 0, 1)
        else:
            target_h = target_h // 8
            target_w = target_w // 8
            logger.debug(f"Latent crop size: {(target_h, target_w)}")

        H, W = entry.pixel.shape[-2:]
        if self.use_central_crop:
            dh, dw = (H - target_h) // 2, (W - target_w) // 2 # crop 尺寸也使用 bucket 尺寸
        else:
            assert H >= target_h and W >= target_w, f"{H}<{target_h} or {W}<{target_w}"
            dh, dw = random.randint(0, H - target_h), random.randint(0, W - target_w)

        entry.pixel = entry.pixel[:, dh : dh + target_h, dw : dw + target_w] # crop 尺寸也使用 bucket 尺寸
        logger.debug(f"Cropped to shape: {entry.pixel.shape}, target shape: {(target_h, target_w)}")
        return entry, dh, dw

    def generate_buckets(self):
        pass

    def assign_buckets(self):
        img_res = np.array(self.store.raw_res)
        self.to_size = {}
        self.bucket_content = defaultdict(list)

        # Assign images to buckets
        for idx, (img_width, img_height) in enumerate(img_res):
            img_area = img_width * img_height

            # Check if the image needs to be resized (i.e., only allow downsizing)
            target_width, target_height = img_width, img_height # 初始化为原始尺寸
            if img_area > self.target_area:
                scale_factor = math.sqrt(self.target_area / img_area)
                target_width = math.floor(img_width * scale_factor)
                target_height = math.floor(img_height * scale_factor)

            # 计算目标尺寸（确保是 divisible 的倍数）
            bucket_width = math.floor(target_width / self.divisible) * self.divisible
            bucket_height = math.floor(target_height / self.divisible) * self.divisible

            # 确保最小尺寸
            bucket_width = max(self.divisible, bucket_width)
            bucket_height = max(self.divisible, bucket_height)

            reso = (bucket_height, bucket_width) # 存储为 (height, width)
            self.bucket_content[reso].append(idx)
            self.to_size[idx] = reso

        self.bucket_content = [v for k, v in self.bucket_content.items()]