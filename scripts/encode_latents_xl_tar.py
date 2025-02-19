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

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from dataclasses import dataclass
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Generator, Optional


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
    parser = argparse.ArgumentParser(description="预处理图像并编码潜在表示，支持目录或 tar 文件输入。") #  更详细的描述 (中文)
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="图像的根目录或 tar 文件。" #  修改 help 描述 (中文)
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="保存缓存文件和 dataset.json 的输出目录。") #  更清晰的描述 output (中文)
    parser.add_argument("--no-upscale", "-nu", action="store_true", help="调整大小期间不放大图像。") #  更清晰的 help (中文)
    parser.add_argument("--dtype", "-d", type=str, default="bfloat16", help="潜在表示的数据类型 (float32 或 bfloat16)。") #  更清晰的 help 和可选值 (中文)
    parser.add_argument("--num_workers", "-n", type=int, default=6, help="数据加载器 worker 数量。") #  更清晰的 help (中文)
    parser.add_argument("--metadata_json_path", "-metadata", type=str, default=None, help="存储包含图片 prompt 和 extra 信息的 JSON 文件路径 (可选)。") #  用于存储包含图片prompt和extra的的json (中文)
    parser.add_argument("--use_tar", "-ut", action="store_true", help="启用 tar 文件处理。如果输入是目录并且设置了此标志，则将在目录中搜索并处理 .tar 文件。") #  更详细的 use_tar help (中文)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    root = args.input
    opt = Path(args.output)
    opt.mkdir(exist_ok=True, parents=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    num_workers = args.num_workers
    use_tar = args.use_tar
    metadata_json_path = args.metadata_json_path
    
    vae_path = "stabilityai/sdxl-vae"
    vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=dtype)
    vae.requires_grad_(False)
    vae.eval().cuda()

    if use_tar:
        if not metadata_json_path:
            raise ValueError("--metadata_json_path must be specified when using --use_tar")
        dataset = TarDataset(root, metadata_json_path, dtype=dtype, no_upscale=args.no_upscale)
    else:
        dataset = LatentEncodingDataset(root, dtype=dtype, no_upscale=args.no_upscale)

    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers)

    # 加载或初始化 dataset mapping 用于断点续传
    dataset_json_file = opt / "dataset.json"
    if dataset_json_file.exists():
        with open(dataset_json_file, "r") as f:
            dataset_mapping = json.load(f)
        print(f"Loaded {len(dataset_mapping)} entries from existing dataset mapping.")
    else:
        dataset_mapping = {}

    # 使用单一 cache_full.h5 文件进行缓存
    max_h5_size = 40 * 1024 * 1024 * 1024  # 40GB maximum h5 file size
    cache_filename = "cache_full.h5"
    h5_cache_file = opt / cache_filename
    print(f"Saving cache to {h5_cache_file}")
    file_mode = "w" if not h5_cache_file.exists() else "r+"

    count = 0
    with h5.File(h5_cache_file, file_mode, libver="latest") as F:
        with torch.no_grad():
            for item in tqdm(dataloader):
                if use_tar:
                    img, basepath, prompt, sha1, original_size, dhdw, extra = item
                else:
                    img, basepath, prompt, sha1, original_size, dhdw = item

                if sha1 is None:
                    print(f"\033[33mWarning: {basepath} is invalid. Skipping...\033[0m")
                    continue

                if sha1 in dataset_mapping:
                    count += 1
                    continue

                h, w = original_size
                dataset_mapping[sha1] = {
                    "train_use": True if prompt else False,
                    "train_caption": prompt,
                    "file_path": str(basepath),
                    "train_width": w,
                    "train_height": h,
                }
                if use_tar:
                    dataset_mapping[sha1]["extra"] = extra


                if f"{sha1}.latents" in F:
                    print(f"\033[33mWarning: {str(basepath)} is already cached in h5. Skipping...\033[0m")
                    continue

                img = img.unsqueeze(0).cuda()
                latent = vae.encode(img, return_dict=False)[0]
                latent.deterministic = True
                latent = latent.sample()[0]

                dset = F.create_dataset(
                    f"{sha1}.latents",
                    data=latent.float().cpu().numpy(),
                    compression="gzip",
                )
                dset.attrs["scale"] = False
                dset.attrs["dhdw"] = dhdw


    with open(dataset_json_file, "w") as f:
        json.dump(dataset_mapping, f, indent=4)
    print(f"Dataset processing complete. Total skipped existing images: {count}")

    # 后处理：检查 cache_full.h5 是否超过 40GB，如超过则拆分为多个文件（每个最大 40GB）
    cache_path = opt / cache_filename
    cache_size = cache_path.stat().st_size
    if cache_size > max_h5_size:
        print(f"Cache file {cache_path} size {cache_size} exceeds 40GB. Splitting into multiple h5 files...")
        with h5.File(cache_path, "r", libver="latest") as full_cache:
            keys = list(full_cache.keys())
            split_index = 1
            current_split_size = 0
            split_filename = f"cache_{split_index}.h5"
            split_file_path = opt / split_filename
            new_h5_file = h5.File(split_file_path, "w", libver="latest")
            for key in keys:
                dset = full_cache[key]
                data = dset[...]
                ds_size = data.nbytes  # 数据集大小（字节数）
                
                # 如果累计大小超过限制，则切换到新文件（确保当前文件至少已有一些数据）
                if current_split_size + ds_size > max_h5_size and current_split_size > 0:
                    new_h5_file.close()
                    split_index += 1
                    split_filename = f"cache_{split_index}.h5"
                    split_file_path = opt / split_filename
                    new_h5_file = h5.File(split_file_path, "w", libver="latest")
                    current_split_size = 0
                new_dset = new_h5_file.create_dataset(key, data=data, compression="gzip")
                for attr_key, attr_val in dset.attrs.items():
                    new_dset.attrs[attr_key] = attr_val
                current_split_size += ds_size
            new_h5_file.close()
        print(f"Splitting complete. Generated {split_index} cache files.")