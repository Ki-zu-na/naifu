import argparse
import functools
import hashlib
import math
import cv2
import h5py as h5
import json
import numpy as np
import torch
import tarfile  # 导入 tarfile 模块
import io # 导入 io 模块
import accelerate # Import accelerate
from accelerate.utils import gather_object # Import gather_object

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from dataclasses import dataclass
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Generator, Optional, List, Tuple, Dict, Any # Added more types


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
        print(f"Found {len(self.paths)} images")
        self.dtype = dtype
        self.raw_res = []

        remove_paths = []
        for p in tqdm(
            self.paths,
            desc="Loading image sizes",
            leave=False,
            ascii=True,
        ):
            try:
                w, h = Image.open(p).size
                # 过滤条件：较小边小于 256 或 较大边小于 512 则跳过
                if min(w, h) < 256 or max(w, h) < 512:
                    print(f"\033[33mSkipped image due to low resolution {w}x{h}: {p}\033[0m")
                    remove_paths.append(p)
                    continue
                self.raw_res.append((h, w))
            except Exception as e:
                print(f"\033[33mSkipped: error processing {p}: {e}\033[0m")
                remove_paths.append(p)

        remove_paths = set(remove_paths)
        self.paths = [p for p in self.paths if p not in remove_paths]
        self.length = len(self.raw_res)
        print(f"Loaded {self.length} image sizes")

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
            print(f"\033[31mError processing {self.paths[index]}: {e}\033[0m")
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
        print(f"Found {len(self.tar_files)} tar files")

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
        print(f"Loaded {len(self.image_entries)} images from tar files")


    def _load_metadata(self):
        """加载所有 tar 文件的元数据和指定的 metadata json 文件，并筛选文件名拓展名匹配全局 meta 的图像"""
        global_metadata = {}
        skipped_count = 0  # 添加计数器
        
        if self.metadata_json_path.exists():  # 加载全局 meta json
            print(f"Loading global metadata from {self.metadata_json_path}")
            with open(self.metadata_json_path, 'r') as f:
                global_metadata = json.load(f)
            print(f"Loaded metadata for {len(global_metadata)} images from {self.metadata_json_path}")

        for tar_path in tqdm(self.tar_files, desc="Loading tar metadata", leave=False, ascii=True):
            json_path = tar_path.with_suffix(".json")
            if not json_path.exists():
                print(f"\033[33mWarning: JSON metadata file {json_path} not found for {tar_path}. Skipping tar file.\033[0m")
                continue

            try:
                with open(json_path, 'r') as f:
                    tar_metadata = json.load(f)
                tar_hash = tar_metadata.get("hash")  # 使用 tar 文件的 hash 作为标识
                if not tar_hash:
                    print(f"\033[33mWarning: 'hash' not found in {json_path} for {tar_path}. Skipping tar file.\033[0m")
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
                        print(f"\033[33mWarning: 'sha256' not found for {filename} in {json_path}. Skipping image.\033[0m")
                        continue

                    if sha256 in self.image_metadatas:
                        print(f"\033[33mWarning: Duplicate sha256 {sha256} found. Skipping duplicate entry.\033[0m")
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
                print(f"\033[31mError decoding JSON in {json_path}: {e}. Skipping tar file.\033[0m")
            except KeyError as e:
                print(f"\033[31mKeyError: {e} in {json_path}. Skipping tar file.\033[0m")
            except Exception as e:
                print(f"\033[31mError processing tar metadata {json_path}: {e}. Skipping tar file.\033[0m")

        if skipped_count > 0:  # 在处理完所有文件后输出统计信息
            print(f"\033[33mSkipped {skipped_count} files that do not exist in global meta json.\033[0m")


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
        valid_image_entries = []
        img_res = []
        for sha256 in self.image_entries:
            tar_info = self.image_metadatas[sha256]
            try:
                with tarfile.open(tar_info["tar_path"], 'r') as tar:
                    with tar.fileobj as tar_fileobj:  # 使用 tar_fileobj
                        tar_fileobj.seek(tar_info["offset"])  # 使用 offset 定位
                        image_data = tar_fileobj.read(tar_info["size"])  # 读取 size 大小的数据
                        fileobj = io.BytesIO(image_data)  # 将数据转换为文件对象
                        _img = Image.open(fileobj)  # 从内存文件对象中打开图像
                        width, height = _img.size
                        # 新增过滤低分辨率，低于 512x512 的图像直接跳过
                        if width * height < 512 * 512:
                            print(f"\033[33mSkipped image due to low resolution {width}x{height}: {tar_info['filename']} from {tar_info['tar_path']}\033[0m")
                            continue
                        img_res.append((height, width))  # 统一存储为 (h, w)
                        valid_image_entries.append(sha256)
            except Exception as e:
                print(f"\033[31mError loading image size for {tar_info['filename']} from {tar_info['tar_path']}: {e}\033[0m")
                continue
        self.image_entries = valid_image_entries
        if len(img_res) == 0:
            print("\033[31mNo valid images found after filtering by resolution.\033[0m")
            img_res = np.array([])
        else:
            img_res = np.array(img_res)
        
        self.bucket_content = [[] for _ in range(len(self.buckets_sizes))]
        self.to_ratio = {}

        # Assign images to buckets
        for idx, img_ratio in enumerate(img_res[:, 0] / img_res[:, 1]):
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
            print(f"\033[31mError processing {tar_info['filename']} from {tar_info['tar_path']}: {e}\033[0m")
            return None, str(tar_info["tar_path"]), None, None, None, None, None

        return img, f"{tar_info['tar_path']}@{tar_info['filename']}", prompt, new_sha1, original_size, dhdw, extra


    def __len__(self):
        return len(self.image_entries)


def get_args():
    parser = argparse.ArgumentParser(description="使用 accelerate 预处理图像并编码潜在表示，支持目录或 tar 文件输入。") # Updated description
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="图像的根目录或包含 tar 文件的目录。" # Clarified help text
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="保存缓存文件和 dataset.json 的输出目录。")
    parser.add_argument("--vae_path", type=str, default="stabilityai/sdxl-vae", help="预训练 VAE 模型的路径或 huggingface repo id。") # Added VAE path argument
    parser.add_argument("--no-upscale", "-nu", action="store_true", help="调整大小期间不放大图像。")
    parser.add_argument("--dtype", "-d", type=str, default="bfloat16", help="潜在表示的数据类型 (float32 或 bfloat16)。")
    parser.add_argument("--num_workers", "-n", type=int, default=6, help="数据加载器 worker 数量。")
    parser.add_argument("--batch_size", "-bs", type=int, default=1, help="每个 GPU 的批处理大小。") # Added batch size argument
    parser.add_argument("--metadata_json_path", "-metadata", type=str, default=None, help="存储包含图片 prompt 和 extra 信息的 JSON 文件路径 (可选)。")
    parser.add_argument("--use_tar", "-ut", action="store_true", help="启用 tar 文件处理。如果输入是目录并且设置了此标志，则将在目录中搜索并处理 .tar 文件。")
    return parser.parse_args()

# Custom collate function to handle None values from dataset __getitem__
def collate_fn(batch: List[Optional[Tuple[torch.Tensor, str, str, str, tuple[int, int], tuple[int, int], Dict[str, Any]]]]) -> Optional[Dict[str, Any]]:
    # Filter out None items (errors during processing)
    batch = [item for item in batch if item is not None and item[0] is not None]
    if not batch:
        return None # Return None if the whole batch failed

    imgs, basepaths, prompts, sha1s, original_sizes, dhdws, extras = zip(*batch)

    # Stack tensors and package other data into lists
    return {
        "imgs": torch.stack(imgs),
        "basepaths": list(basepaths),
        "prompts": list(prompts),
        "sha1s": list(sha1s),
        "original_sizes": list(original_sizes),
        "dhdws": list(dhdws),
        "extras": list(extras) if args.use_tar else None # Only include extras if using tar dataset
    }


if __name__ == "__main__":
    args = get_args()

    # Initialize Accelerate
    accelerator = accelerate.Accelerator()
    device = accelerator.device # Get the device for the current process

    opt = Path(args.output)
    # Create output directory only on the main process
    if accelerator.is_main_process:
        opt.mkdir(exist_ok=True, parents=True)

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    # Load VAE model (no need to move to device explicitly, accelerator.prepare will handle it)
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(dtype=dtype) # Load with specified dtype
    vae.requires_grad_(False)
    vae.eval()

    # Create dataset
    if args.use_tar:
        dataset = TarDataset(args.input, args.metadata_json_path, dtype=dtype, no_upscale=args.no_upscale)
    else:
        # Ensure LatentEncodingDataset also returns the 'extra' field (even if empty) for consistency
        # Modify LatentEncodingDataset.__getitem__ if needed, or adjust collate_fn
        # For now, assuming LatentEncodingDataset is adjusted or collate_fn handles the difference.
        # If LatentEncodingDataset doesn't return 'extra', the collate_fn needs adjustment.
        # Let's assume LatentEncodingDataset returns 7 items like TarDataset for simplicity here.
        # A safer approach would be conditional logic in collate_fn or __main__.
        dataset = LatentEncodingDataset(args.input, dtype=dtype, no_upscale=args.no_upscale)
        # Quick patch to make LatentEncodingDataset return 7 items for collate_fn consistency
        original_getitem = dataset.__getitem__
        def patched_getitem(index):
            result = original_getitem(index)
            if result[0] is None: # Handle error case
                 return (None,) * 7 # Return tuple of Nones with correct length
            img, basepath, prompt, sha1, original_size, dhdw = result
            return img, basepath, prompt, sha1, original_size, dhdw, {} # Add empty dict for extra
        dataset.__getitem__ = patched_getitem


    # Create DataLoader (no sampler needed, accelerator handles data distribution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, # Use batch_size argument
        num_workers=args.num_workers,
        collate_fn=collate_fn, # Use custom collate_fn
        shuffle=False # Usually false for encoding tasks
    )

    # Prepare model and dataloader with accelerator
    vae, dataloader = accelerator.prepare(vae, dataloader)

    # HDF5 file and dataset mapping handling (only on main process)
    h5_cache_file = opt / "cache_latents.h5"
    dataset_json_file = opt / "dataset.json"
    dataset_mapping = {}
    processed_sha1s = set() # Keep track of processed items to avoid duplicates when gathering

    # Load existing data if resuming (only on main process)
    if accelerator.is_main_process:
        if dataset_json_file.exists():
            print(f"Loading existing dataset mapping from {dataset_json_file}")
            with open(dataset_json_file, "r") as f:
                dataset_mapping = json.load(f)
                processed_sha1s.update(dataset_mapping.keys())
                print(f"Loaded {len(dataset_mapping)} existing entries.")

        # Open HDF5 file in append mode if it exists, otherwise write mode
        file_mode = "a" if h5_cache_file.exists() else "w"
        h5_cache = h5.File(h5_cache_file, file_mode, libver="latest")
        # Load existing keys from HDF5 to prevent reprocessing
        processed_sha1s.update([key.replace(".latents","") for key in h5_cache.keys()])
        print(f"Found {len(processed_sha1s)} processed SHA1s (JSON + HDF5).")

    else:
        h5_cache = None # Only main process interacts with the file

    # Process batches
    num_processed = 0
    num_skipped = 0
    with torch.no_grad():
        # Wrap dataloader with accelerator.main_process_first for tqdm on main process only
        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process, desc="Encoding Latents")
        for batch in progress_bar:
            if batch is None: # Skip if collate_fn returned None (entire batch failed)
                 continue

            # Data is already on the correct device thanks to accelerator.prepare(dataloader)
            imgs = batch["imgs"]
            basepaths = batch["basepaths"]
            prompts = batch["prompts"]
            sha1s = batch["sha1s"]
            original_sizes = batch["original_sizes"]
            dhdws = batch["dhdws"]
            extras = batch["extras"] # Will be None if not args.use_tar or if dataset doesn't provide it

            # Filter out items already processed (checking sha1s before encoding)
            valid_indices = [i for i, sha1 in enumerate(sha1s) if sha1 not in processed_sha1s]

            if not valid_indices:
                num_skipped += len(sha1s)
                if accelerator.is_main_process:
                     progress_bar.set_postfix({"skipped": num_skipped, "processed": num_processed})
                continue # Skip batch if all items are already processed

            # Select only the valid items to encode
            imgs_to_encode = imgs[valid_indices]
            valid_sha1s = [sha1s[i] for i in valid_indices]
            valid_basepaths = [basepaths[i] for i in valid_indices]
            valid_prompts = [prompts[i] for i in valid_indices]
            valid_original_sizes = [original_sizes[i] for i in valid_indices]
            valid_dhdws = [dhdws[i] for i in valid_indices]
            valid_extras = [extras[i] for i in valid_indices] if extras else [{}] * len(valid_indices) # Handle extras

            # Encode valid images
            # Use .module if model was wrapped by accelerator (e.g., DDP)
            model_to_encode = accelerator.unwrap_model(vae)
            latent_dist = model_to_encode.encode(imgs_to_encode).latent_dist
            # latent_dist = vae.encode(imgs_to_encode).latent_dist # Original line - might fail if wrapped
            latents = latent_dist.sample() * model_to_encode.config.scaling_factor # Apply scaling factor

            # Gather results from all processes
            gathered_latents = accelerator.gather(latents)
            gathered_sha1s = gather_object(valid_sha1s)
            gathered_basepaths = gather_object(valid_basepaths)
            gathered_prompts = gather_object(valid_prompts)
            gathered_original_sizes = gather_object(valid_original_sizes)
            gathered_dhdws = gather_object(valid_dhdws)
            gathered_extras = gather_object(valid_extras)


            # Write results and update mapping (only on main process)
            if accelerator.is_main_process:
                for i in range(len(gathered_sha1s)):
                    sha1 = gathered_sha1s[i]
                    if sha1 in processed_sha1s: # Double check after gathering
                        num_skipped +=1
                        continue

                    latent_np = gathered_latents[i].float().cpu().numpy()
                    basepath = gathered_basepaths[i]
                    prompt = gathered_prompts[i]
                    h, w = gathered_original_sizes[i]
                    dhdw = gathered_dhdws[i]
                    extra = gathered_extras[i]


                    # Save to HDF5
                    dset = h5_cache.create_dataset(
                        f"{sha1}.latents",
                        data=latent_np,
                        compression="gzip",
                    )
                    # dset.attrs["scale"] = False # Scaling is now applied before saving
                    dset.attrs["dhdw"] = dhdw # Store dhdw crop info

                    # Update mapping
                    dataset_mapping[sha1] = {
                        "train_use": bool(prompt), # Use bool() for clarity
                        "train_caption": prompt,
                        "file_path": str(basepath),
                        "train_width": w,
                        "train_height": h,
                    }
                    if args.use_tar and extra: # Add extra metadata if available
                        dataset_mapping[sha1]["extra"] = extra

                    processed_sha1s.add(sha1) # Mark as processed
                    num_processed += 1

                # Update progress bar postfix
                progress_bar.set_postfix({"skipped": num_skipped, "processed": num_processed})

            # Ensure all processes wait here before the next batch,
            # especially important if file writing takes time.
            accelerator.wait_for_everyone()


    # Final save and cleanup (only on main process)
    if accelerator.is_main_process:
        print(f"Processed {num_processed} new images.")
        print(f"Skipped {num_skipped} previously processed or invalid images.")

        # Save final dataset mapping
        print(f"Saving final dataset mapping to {dataset_json_file}...")
        with open(dataset_json_file, "w") as f:
            json.dump(dataset_mapping, f, indent=4)
        print("Dataset mapping saved.")

        # Close HDF5 file
        if h5_cache:
            h5_cache.close()
            print(f"HDF5 cache saved to {h5_cache_file}")

    # Wait for the main process to finish writing before exiting
    accelerator.wait_for_everyone()
    print("Processing finished.")