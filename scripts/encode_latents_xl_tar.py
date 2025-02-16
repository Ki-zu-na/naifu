import argparse
import functools
import hashlib
import math
import cv2
import h5py as h5
import json
import numpy as np
import torch
import tarfile
import io
import os
import socket

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from dataclasses import dataclass
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Generator, Optional

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def setup_distributed():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", None)
        if master_port is None:
            master_port = str(get_free_port())
            os.environ["MASTER_PORT"] = master_port
        init_method = f"tcp://{master_addr}:{master_port}"
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
        return rank, world_size
    else:
        return 0, 1

def get_sha1(path: Path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()

image_suffix = set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"])

def is_img(path: Path):
    return path.suffix in image_suffix

@dataclass
class Entry:
    is_latent: bool
    pixel: torch.Tensor

def dirwalk(path: Path, cond: Optional[Callable] = None) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, cond)
        else:
            if isinstance(cond, Callable):
                if not cond(p):
                    continue
            yield p

class LatentEncodingDataset(Dataset):
    def __init__(self, root: str | Path, dtype=torch.float32, no_upscale=False):
        self.tr = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.paths = sorted(list(dirwalk(Path(root), is_img)))
        print(f"Rank {rank}: Found {len(self.paths)} images") #  显示 Rank 信息
        self.dtype = dtype
        self.raw_res = []

        remove_paths = []
        for p in tqdm(
            self.paths,
            desc=f"Rank {rank}: Loading image sizes", #  显示 Rank 信息
            leave=False,
            ascii=True,
            disable=rank!=0 #  仅 rank 0 显示进度条
        ):
            try:
                w, h = Image.open(p).size
                self.raw_res.append((h, w))
            except Exception as e:
                print(f"\033[33mRank {rank}: Skipped: error processing {p}: {e}\033[0m") #  显示 Rank 信息
                remove_paths.append(p)

        remove_paths = set(remove_paths)
        self.paths = [p for p in self.paths if p not in remove_paths]
        self.length = len(self.raw_res)
        print(f"Rank {rank}: Loaded {self.length} image sizes") #  显示 Rank 信息

        self.fit_bucket_func = self.fit_bucket
        if no_upscale:
            self.fit_bucket_func = self.fit_bucket_no_upscale

        self.target_area = 1536 * 1536
        self.max_size, self.min_size, self.divisible = 4096, 512, 64
        self.generate_buckets()
        self.assign_buckets()

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
        resos.add(
            ((int(np.sqrt(self.target_area)) // self.divisible) * self.divisible,) * 2
        )
        self.buckets_sizes = np.array(sorted(resos))
        self.bucket_ratios = self.buckets_sizes[:, 0] / self.buckets_sizes[:, 1]
        self.ratio_to_bucket = {
            ratio: hw for ratio, hw in zip(self.bucket_ratios, self.buckets_sizes)
        }

    def assign_buckets(self):
        img_res = np.array(self.raw_res)
        img_ratios = img_res[:, 0] / img_res[:, 1]
        self.bucket_content = [[] for _ in range(len(self.buckets_sizes))]
        self.to_ratio = {}

        # Assign images to buckets
        for idx, img_ratio in enumerate(img_ratios):
            diff = np.abs(self.bucket_ratios - img_ratio)
            bucket_idx = np.argmin(diff)
            self.bucket_content[bucket_idx].append(idx)
            self.to_ratio[idx] = self.bucket_ratios[bucket_idx]

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

    @torch.no_grad()
    def fit_bucket(self, idx, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        base_ratio = h / w
        target_ratio = self.to_ratio[idx]
        target_h, target_w = self.ratio_to_bucket[target_ratio]
        resize_h, resize_w = self.fit_dimensions(base_ratio, target_h, target_w)
        interp = cv2.INTER_AREA if resize_h < h else cv2.INTER_LANCZOS4
        img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)

        dh, dw = abs(target_h - img.shape[0]) // 2, abs(target_w - img.shape[1]) // 2
        img = img[dh : dh + target_h, dw : dw + target_w]
        return img, (dh, dw)

    @torch.no_grad()
    def fit_bucket_no_upscale(self, idx, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        img_area = h * w

        # Check if the image needs to be resized (i.e., only allow downsizing)
        if img_area > self.target_area:
            scale_factor = math.sqrt(self.target_area / img_area)
            resize_w = math.floor(w * scale_factor / self.divisible) * self.divisible
            resize_h = math.floor(h * scale_factor / self.divisible) * self.divisible
        else:
            resize_w, resize_h = w, h

        target_w = resize_w - resize_w % self.divisible
        target_h = resize_h - resize_h % self.divisible

        interp = cv2.INTER_AREA if resize_h < h else cv2.INTER_LANCZOS4
        img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)

        dh, dw = abs(target_h - img.shape[0]) // 2, abs(target_w - img.shape[1]) // 2
        img = img[dh : dh + target_h, dw : dw + target_w]
        return img, (dh, dw)

    def __getitem__(self, index) -> tuple[torch.Tensor, str, str, str, tuple[int, int], tuple[int, int]]: # 修改返回值类型注解
        try:
            img_path = self.paths[index]
            _img = Image.open(img_path)
            with img_path.with_suffix(".txt").open("r") as f: # 假设prompt文件是同名.txt
                prompt = f.read()

            if _img.mode == "RGB":
                img = np.array(_img)
            elif _img.mode == "RGBA":
                # transparent images
                baimg = Image.new('RGB', _img.size, (255, 255, 255))
                baimg.paste(_img, (0, 0), _img)
                img = np.array(baimg)
            else:
                img = np.array(_img.convert("RGB"))

            original_size = img.shape[:2]
            img, dhdw = self.fit_bucket_func(index, img)
            img = self.tr(img).to(self.dtype)
            sha1 = get_sha1(img_path)
        except Exception as e:
            print(f"\033[31mRank {rank}: Error processing {self.paths[index]}: {e}\033[0m") #  显示 Rank 信息
            return None, str(self.paths[index]), None, None, None, None # 修改错误返回类型
        return img, str(self.paths[index]), prompt, sha1, original_size, dhdw # 修改返回值类型

    def __len__(self):
        return len(self.paths)


class TarDataset(Dataset):
    def __init__(self, tar_dir: str | Path, metadata_json_path: str | Path, dtype=torch.float32, no_upscale=False):
        self.tar_dir = Path(tar_dir)
        self.metadata_json_path = Path(metadata_json_path)
        self.dtype = dtype
        self.no_upscale = no_upscale
        self.tr = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.tar_files = sorted([p for p in dirwalk(self.tar_dir) if p.suffix == ".tar"]) # 查找所有 tar 文件
        if not self.tar_files:
            raise ValueError(f"No tar files found in {tar_dir}")
        print(f"Rank {rank}: Found {len(self.tar_files)} tar files") #  显示 Rank 信息

        #  分割 tar 文件列表
        if world_size > 1:
            self.tar_files = self.tar_files[rank::world_size]
            print(f"Rank {rank}: Processing {len(self.tar_files)} tar files out of {len(sorted([p for p in dirwalk(self.tar_dir) if p.suffix == '.tar']))} total.") #  显示分割后的 tar 文件数量

        self.image_metadatas = {} # 存储所有图像的元数据，key 是 sha1
        self.image_entries = [] # 存储图像条目，每个条目包含 tar 文件路径，图像文件名，以及 sha1
        self._load_metadata() # 加载元数据

        self.fit_bucket_func = self.fit_bucket
        if no_upscale:
            self.fit_bucket_func = self.fit_bucket_no_upscale

        self.target_area = 1536 * 1536
        self.max_size, self.min_size, self.divisible = 4096, 512, 64
        self.generate_buckets()
        self.assign_buckets()
        print(f"Rank {rank}: Loaded {len(self.image_entries)} images from tar files") #  显示 Rank 信息


    def _load_metadata(self):
        """加载所有 tar 文件的元数据和指定的 metadata json 文件，并筛选文件名拓展名匹配全局 meta 的图像"""
        global_metadata = {}
        skipped_count = 0  # 添加计数器

        if self.metadata_json_path.exists():  # 加载全局 meta json
            print(f"Rank {rank}: Loading global metadata from {self.metadata_json_path}") #  显示 Rank 信息
            with open(self.metadata_json_path, 'r') as f:
                global_metadata = json.load(f)
            print(f"Rank {rank}: Loaded metadata for {len(global_metadata)} images from {self.metadata_json_path}") #  显示 Rank 信息

        for tar_path in tqdm(
            self.tar_files,
            desc=f"Rank {rank}: Loading tar metadata", #  显示 Rank 信息
            leave=False,
            ascii=True,
            disable=rank!=0 #  仅 rank 0 显示进度条
        ):
            json_path = tar_path.with_suffix(".json")
            if not json_path.exists():
                print(f"\033[33mRank {rank}: Warning: JSON metadata file {json_path} not found for {tar_path}. Skipping tar file.\033[0m") #  显示 Rank 信息
                continue

            try:
                with open(json_path, 'r') as f:
                    tar_metadata = json.load(f)
                tar_hash = tar_metadata.get("hash")  # 使用 tar 文件的 hash 作为标识
                if not tar_hash:
                    print(f"\033[33mRank {rank}: Warning: 'hash' not found in {json_path} for {tar_path}. Skipping tar file.\033[0m") #  显示 Rank 信息
                    continue

                for filename, file_info in tar_metadata["files"].items():
                    # 若文件扩展名不匹配图像类型则跳过
                    if not is_img(Path(filename)):
                        continue

                    # 检查全局 meta json 中是否存在该文件名
                    filename_key = Path(filename).name
                    if global_metadata and filename_key not in global_metadata:
                        skipped_count += 1  # 增加计数
                        continue

                    sha256 = file_info.get("sha256")
                    if not sha256:
                        print(f"\033[33mRank {rank}: Warning: 'sha256' not found for {filename} in {json_path}. Skipping image.\033[0m") #  显示 Rank 信息
                        continue

                    if sha256 in self.image_metadatas:
                        print(f"\033[33mRank {rank}: Warning: Duplicate sha256 {sha256} found. Skipping duplicate entry.\033[0m") #  显示 Rank 信息
                        continue

                    self.image_metadatas[sha256] = {  # 存储图像元数据
                        "tar_path": tar_path,
                        "offset": file_info["offset"],
                        "size": file_info["size"],
                        "filename": filename,
                        "sha256": sha256,
                        "tar_hash": tar_hash  # 记录 tar hash
                    }
                    # 尝试从 global_metadata 中获取 caption 和 extra 信息
                    global_meta_entry = global_metadata.get(filename_key)
                    if global_meta_entry:
                        self.image_metadatas[sha256]["caption"] = global_meta_entry.get("tag_string_general", "")
                        extra_keys = [
                            "tag_string_general", "tag_string_character", "tag_string_copyright",
                            "tag_string_artist", "tag_string_meta", "score", "regular_summary", "individual_parts",
                            "midjourney_style_summary", "deviantart_commission_request", "brief_summary", "rating", "aes_rating"
                        ]
                        self.image_metadatas[sha256]["extra"] = {k: global_meta_entry.get(k, "") for k in extra_keys}
                    else:
                        self.image_metadatas[sha256]["caption"] = ""
                        self.image_metadatas[sha256]["extra"] = {}

                    self.image_entries.append(sha256)

            except json.JSONDecodeError as e:
                print(f"\033[31mRank {rank}: Error decoding JSON in {json_path}: {e}. Skipping tar file.\033[0m") #  显示 Rank 信息
            except KeyError as e:
                print(f"\033[31mRank {rank}: KeyError: {e} in {json_path}. Skipping tar file.\033[0m") #  显示 Rank 信息
            except Exception as e:
                print(f"\033[31mRank {rank}: Error processing tar metadata {json_path}: {e}. Skipping tar file.\033[0m") #  显示 Rank 信息

        if skipped_count > 0:  # 在处理完所有文件后输出统计信息
            print(f"\033[33mRank {rank}: Skipped {skipped_count} files that do not exist in global meta json.\033[0m") #  显示 Rank 信息


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
        resos.add(
            ((int(np.sqrt(self.target_area)) // self.divisible) * self.divisible,) * 2
        )
        self.buckets_sizes = np.array(sorted(resos))
        self.bucket_ratios = self.buckets_sizes[:, 0] / self.buckets_sizes[:, 1]
        self.ratio_to_bucket = {
            ratio: hw for ratio, hw in zip(self.bucket_ratios, self.buckets_sizes)
        }

    def assign_buckets(self):
        img_res = []
        for sha256 in self.image_entries:
            try:
                tar_info = self.image_metadatas[sha256]
                with tarfile.open(tar_info["tar_path"], 'r') as tar:
                    with tar.fileobj as tar_fileobj: #  使用 tar_fileobj
                        tar_fileobj.seek(tar_info["offset"]) #  使用 offset 定位
                        image_data = tar_fileobj.read(tar_info["size"]) #  读取 size 大小的数据
                        fileobj = io.BytesIO(image_data) #  使用 io.BytesIO 将数据转换为文件对象
                        _img = Image.open(fileobj) # 从内存文件对象中打开图像
                        img_res.append(_img.size[::-1]) # 注意 PIL 返回 (width, height), 这里需要 (height, width)
            except Exception as e:
                print(f"\033[31mRank {rank}: Error loading image size for {tar_info['filename']} from {tar_info['tar_path']}: {e}\033[0m") #  显示 Rank 信息
                img_res.append((1024, 1024)) # 错误时默认分辨率

        img_res = np.array(img_res)
        img_ratios = img_res[:, 0] / img_res[:, 1]
        self.bucket_content = [[] for _ in range(len(self.buckets_sizes))]
        self.to_ratio = {}

        # Assign images to buckets
        for idx, img_ratio in enumerate(img_ratios):
            diff = np.abs(self.bucket_ratios - img_ratio)
            bucket_idx = np.argmin(diff)
            self.bucket_content[bucket_idx].append(idx)
            self.to_ratio[idx] = self.bucket_ratios[bucket_idx]


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

    @torch.no_grad()
    def fit_bucket(self, idx, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        base_ratio = h / w
        target_ratio = self.to_ratio[idx]
        target_h, target_w = self.ratio_to_bucket[target_ratio]
        resize_h, resize_w = self.fit_dimensions(base_ratio, target_h, target_w)
        interp = cv2.INTER_AREA if resize_h < h else cv2.INTER_LANCZOS4
        img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)

        dh, dw = abs(target_h - img.shape[0]) // 2, abs(target_w - img.shape[1]) // 2
        img = img[dh : dh + target_h, dw : dw + target_w]
        return img, (dh, dw)

    @torch.no_grad()
    def fit_bucket_no_upscale(self, idx, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        img_area = h * w

        # Check if the image needs to be resized (i.e., only allow downsizing)
        if img_area > self.target_area:
            scale_factor = math.sqrt(self.target_area / img_area)
            resize_w = math.floor(w * scale_factor / self.divisible) * self.divisible
            resize_h = math.floor(h * scale_factor / self.divisible) * self.divisible
        else:
            resize_w, resize_h = w, h

        target_w = resize_w - resize_w % self.divisible
        target_h = resize_h - resize_h % self.divisible

        interp = cv2.INTER_AREA if resize_h < h else cv2.INTER_LANCZOS4
        img = cv2.resize(img, (resize_w, resize_h), interpolation=interp)

        dh, dw = abs(target_h - img.shape[0]) // 2, abs(target_w - img.shape[1]) // 2
        img = img[dh : dh + target_h, dw : dw + target_w]
        return img, (dh, dw)

    @torch.no_grad()
    def __getitem__(self, index) -> tuple[torch.Tensor, str, str, str, tuple[int, int], tuple[int, int], dict]:
        sha256 = self.image_entries[index]
        tar_info = self.image_metadatas[sha256]
        try:
            with tarfile.open(tar_info["tar_path"], 'r') as tar:
                with tar.fileobj as tar_fileobj:
                    tar_fileobj.seek(tar_info["offset"])
                    image_data = tar_fileobj.read(tar_info["size"])
                    fileobj = io.BytesIO(image_data)
                    _img = Image.open(fileobj)
                    if _img.mode == "RGB":
                        img = np.array(_img)
                    elif _img.mode == "RGBA":
                        baimg = Image.new("RGB", _img.size, (255, 255, 255))
                        baimg.paste(_img, (0, 0), _img)
                        img = np.array(baimg)
                    else:
                        img = np.array(_img.convert("RGB"))
                    original_size = img.shape[:2]
                    img, dhdw = self.fit_bucket_func(index, img)
                    img = self.tr(img).to(self.dtype)
                    prompt = tar_info.get("caption", "")
                    extra = tar_info.get("extra", {})
                    new_sha1 = hashlib.sha1(image_data).hexdigest()
        except Exception as e:
            print(f"\033[31mRank {rank}: Error processing {tar_info['filename']} from {tar_info['tar_path']}: {e}\033[0m") #  显示 Rank 信息
            return None, str(tar_info["tar_path"]), None, None, None, None, None

        return img, f"{tar_info['tar_path']}@{tar_info['filename']}", prompt, new_sha1, original_size, dhdw, extra


    def __len__(self):
        return len(self.image_entries)


def get_args():
    parser = argparse.ArgumentParser(description="预处理图像并编码潜在表示，支持目录或 tar 文件输入。") #  更详细的描述 (中文)
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="图像的根目录或 tar 文件。" #  修改 help 描述 (中文)
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="保存缓存文件和 dataset.json 的输出目录。") #  更清晰的描述 output (中文)
    parser.add_argument("--no-upscale", "-nu", action="store_true", help="调整大小期间不放大图像。") #  更清晰的 help (中文)
    parser.add_argument("--dtype", "-d", type=str, default="bfloat16", help="潜在表示的数据类型 (float32 或 bfloat16)。") #  更清晰的 help 和可选值 (中文)
    parser.add_argument("--num_workers", "-nw", type=int, default=12, help="数据加载器 worker 数量。") #  更清晰的 help (中文)
    parser.add_argument("--metadata_json_path", "-metadata", type=str, default=None, help="存储包含图片 prompt 和 extra 信息的 JSON 文件路径。") #  用于存储包含图片prompt和extra的的json
    parser.add_argument("--use_tar", "-ut", action="store_true", help="启用 tar 文件处理。如果输入是目录并且设置了此标志，则将在目录中搜索并处理 .tar 文件。") #  更详细的 use_tar help (中文)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    rank, world_size = setup_distributed() #  初始化分布式环境
    args = get_args()
    root = args.input
    opt = Path(args.output)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    num_workers = args.num_workers
    metadata_json_path = args.metadata_json_path
    use_tar = args.use_tar

    vae_path = "stabilityai/sdxl-vae"
    vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=dtype)
    vae.requires_grad_(False)
    vae.eval().cuda()

    if use_tar:
        dataset = TarDataset(root, dtype=dtype, no_upscale=args.no_upscale, metadata_json_path=metadata_json_path)
    else:
        dataset = LatentEncodingDataset(root, dtype=dtype, no_upscale=args.no_upscale)

    #  使用 DistributedSampler，如果需要
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers, sampler=sampler)
    opt.mkdir(exist_ok=True, parents=True)
    assert opt.is_dir(), f"{opt} is not a directory"

    cache_filename_prefix = "cache" #  缓存文件名前缀
    dataset_mapping = {}
    max_h5_size = 40 * 1024 * 1024 * 1024 #  40GB

    h5_file_list = []
    current_file_index = 0

    def create_new_h5_file(opt, current_file_index, h5_file_list, rank): #  添加 rank 参数
        new_filename = f"{cache_filename_prefix}_rank{rank}_part_{current_file_index:03d}.h5" #  文件名包含 rank
        new_h5_cache_file = opt / new_filename
        h5_file_list.append(new_h5_cache_file)
        print(f"Rank {rank}: Creating new cache file: {new_h5_cache_file}") #  显示 Rank 信息
        new_f = h5.File(new_h5_cache_file, "w", libver="latest")
        return new_f, new_h5_cache_file

    f, h5_cache_file = create_new_h5_file(opt, current_file_index, h5_file_list, rank)

    with torch.no_grad():
        for item in tqdm(dataloader, desc=f"Rank {rank}: Encoding latents", leave=False, ascii=True, disable=rank!=0): #  显示 Rank 信息，仅 rank 0 显示进度条
            if use_tar:
                img, basepath, prompt, sha1, original_size, dhdw, extra = item
            else:
                img, basepath, prompt, sha1, original_size, dhdw = item

            if sha1 is None:
                print(f"\033[33mRank {rank}: Warning: {basepath} is invalid. Skipping...\033[0m") #  显示 Rank 信息
                continue

            dataset_mapping[sha1] = {
                "train_use": True if prompt else False,
                "train_caption": prompt,
                "file_path": str(basepath),
                "train_width": original_size[1], #  修正为 original_size[1]
                "train_height": original_size[0], #  修正为 original_size[0]
            }
            if use_tar:
                dataset_mapping[sha1]["extra"] = extra

            if f"{sha1}.latents" in f:
                print(f"\033[33mRank {rank}: Warning: {str(basepath)} is already cached. Skipping...\033[0m") #  显示 Rank 信息
                continue

            img = img.unsqueeze(0).cuda()
            latent = vae.encode(img, return_dict=False)[0]
            latent.deterministic = True
            latent = latent.sample()[0]
            d = f.create_dataset(
                f"{sha1}.latents",
                data=latent.float().cpu().numpy(),
                compression="gzip",
            )
            d.attrs["scale"] = False
            d.attrs["dhdw"] = dhdw

            if f.id.get_filesize() > max_h5_size:
                f.close()
                current_file_index += 1
                f, h5_cache_file = create_new_h5_file(opt, current_file_index, h5_file_list, rank)

    f.close() #  关闭最后一个 h5 文件

    #  保存 dataset.json，文件名包含 rank
    dataset_json_file = opt / f"dataset_rank{rank}.json"
    with open(dataset_json_file, "w") as f_json:
        json.dump(dataset_mapping, f_json, indent=4)
    print(f"Rank {rank}: Dataset mapping saved to {dataset_json_file}") #  显示 Rank 信息

    if  rank == 0: #  仅 rank 0 执行合并操作
        print("Rank 0: Starting cache merging...")
        merged_dataset_mapping = {}
        merged_h5_file = h5.File(opt / "cache_merged.h5", 'w', libver='latest') #  合并后的 h5 文件名
        merged_h5_file_list = [opt / "cache_merged.h5"] #  合并后的 h5 文件列表
        current_merged_file_index = 0

        #  收集所有 rank 的 dataset.json 文件并合并
        for r in range(world_size):
            dataset_json_path = opt / f"dataset_rank{r}.json"
            if dataset_json_path.exists():
                with open(dataset_json_path, 'r') as f_json:
                    rank_dataset_mapping = json.load(f_json)
                    merged_dataset_mapping.update(rank_dataset_mapping)
                print(f"Rank 0: Merged dataset mapping from {dataset_json_path}") #  显示 Rank 信息
            else:
                print(f"\033[33mRank 0: Warning: Dataset JSON file {dataset_json_path} not found.\033[0m") #  显示 Rank 信息

        #  合并 dataset.json
        merged_dataset_json_file = opt / "dataset.json"
        with open(merged_dataset_json_file, "w") as f_json:
            json.dump(merged_dataset_mapping, f_json, indent=4)
        print(f"Rank 0: Merged dataset mapping saved to {merged_dataset_json_file}") #  显示 Rank 信息

        #  收集所有 rank 的 h5 文件并合并
        rank_h5_files_pattern = opt / f"{cache_filename_prefix}_rank*.h5"
        rank_h5_files = sorted(list(opt.glob(str(rank_h5_files_pattern))))
        print(f"Rank 0: Found {len(rank_h5_files)} rank-specific cache files to merge.") #  显示 Rank 信息

        for rank_h5_file_path in tqdm(rank_h5_files, desc="Rank 0: Merging H5 files", ascii=True): #  合并 h5 文件的进度条
            if rank_h5_file_path.exists():
                with h5.File(rank_h5_file_path, 'r', libver='latest') as rank_h5_file:
                    for key in rank_h5_file.keys():
                        if key in merged_h5_file:
                            print(f"\033[33mRank 0: Warning: Key {key} already exists in merged H5 file. Skipping.\033[0m") #  显示 Rank 信息
                            continue
                        data = rank_h5_file[key][:]
                        attrs = dict(rank_h5_file[key].attrs)
                        dset = merged_h5_file.create_dataset(key, data=data, compression="gzip")
                        dset.attrs.update(attrs)
                        if merged_h5_file.id.get_filesize() > max_h5_size:
                            merged_h5_file.close()
                            current_merged_file_index += 1
                            merged_h5_filename = f"cache_merged_part_{current_merged_file_index:03d}.h5"
                            merged_h5_file = h5.File(opt / merged_h5_filename, 'w', libver='latest')
                            merged_h5_file_list.append(opt / merged_h5_filename)
                print(f"Rank 0: Merged {rank_h5_file_path} into merged cache.") #  显示 Rank 信息
            else:
                print(f"\033[33mRank 0: Warning: Rank-specific H5 file {rank_h5_file_path} not found.\033[0m") #  显示 Rank 信息
        merged_h5_file.close()
        print(f"Rank 0: Merged cache files saved to {[str(p) for p in merged_h5_file_list]}") #  显示 Rank 信息
        #  清理 rank 特定的缓存文件和 dataset.json
        if not args.output.endswith(".tar"): #  如果输出不是 tar 文件，则清理
            for r in range(world_size):
                dataset_json_path = opt / f"dataset_rank{r}.json"
                if dataset_json_path.exists():
                    os.remove(dataset_json_path)
                rank_h5_files_pattern = opt / f"{cache_filename_prefix}_rank{r}_*.h5"
                rank_h5_files = list(opt.glob(str(rank_h5_files_pattern)))
                for rank_h5_file in rank_h5_files:
                    if rank_h5_file.exists():
                        os.remove(rank_h5_file)
            print("Rank 0: Rank-specific cache files and dataset JSONs cleaned up.") #  显示 Rank 信息
        print("Rank 0: Cache merging complete.") #  显示 Rank 信息
    if dist.is_initialized(): #  检查分布式环境是否初始化
        dist.destroy_process_group() #  清理分布式环境