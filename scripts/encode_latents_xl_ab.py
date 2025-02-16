import argparse
import functools
import hashlib
import math
import cv2
import h5py as h5
import json
import numpy as np
import torch
import json
import io
import tarfile

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from dataclasses import dataclass
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Generator, Optional

try:
    import rapidjson as json_lib
except ImportError:
    import json as json_lib

@dataclass
class Entry:
    """
    This class represents an entry in a batch of image data. Each entry contains information about an image and its associated prompt.

    Attributes:
        is_latent (bool): A flag indicating whether the image is in latent space.
        pixel (torch.Tensor): The pixel data of the image.
        prompt (str): The prompt associated with the image.
        extras (dict): A dictionary to store any extra information associated with the image.
    """
    is_latent: bool
    pixel: torch.Tensor
    prompt: str
    extras: dict = None

def load_entry(p: Path, tar_file_handle=None, tar_file_offset=None, tar_file_size=None):
    if tar_file_handle and tar_file_offset is not None and tar_file_size is not None:
        # 从 tar 文件读取图像
        tar_file_obj = tar_file_handle.fileobj
        tar_file_obj.seek(tar_file_offset)
        image_data = tar_file_obj.read(tar_file_size)
        fileobj = io.BytesIO(image_data)
        _img = Image.open(fileobj)
    else:
        # 从文件路径读取图像
        _img = Image.open(p)

    if _img.mode == "RGB":
        img = np.array(_img)
    elif _img.mode == "RGBA":
        baimg = Image.new('RGB', _img.size, (255, 255, 255))
        baimg.paste(_img, (0, 0), _img)
        img = np.array(baimg)
    else:
        img = np.array(_img.convert("RGB"))
    return img

def get_sha1(path: Path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


image_suffix = set([".jpg", ".jpeg", ".png", ".PNG", ".gif", ".bmp", ".tiff", ".tif", ".webp"])


def is_img(path: Path):
    return path.suffix in image_suffix


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
    def __init__(self, root: str | Path, dtype=torch.float32, no_upscale=False, metadata_json_path=None, use_tar=False):
        self.tr = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.root = Path(root)
        self.paths = []
        self.json_data = {}
        self.metadata_json_path = metadata_json_path
        self.prompt_data = {}
        self.tar_file_metas = {} # 用于存储 tar 文件元数据
        self.tar_index_map = {} # 索引到 (tar_path, filename_in_tar) 的映射
        self.is_tar_input = False # 标志是否为 tar 文件输入
        self.tar_file_handle = None # 用于存储打开的 tar 文件句柄
        self.use_tar = use_tar #  保存 use_tar 参数

        if self.metadata_json_path:
            try:
                with open(self.metadata_json_path, 'r') as f:
                    self.prompt_data = json_lib.load(f)
            except FileNotFoundError:
                print(f"\033[33mWarning: Metadata JSON file not found: {self.metadata_json_path}\033[0m")
                self.prompt_data = {}
            except json_lib.JSONDecodeError as e:
                print(f"\033[31mError decoding JSON in {self.metadata_json_path}: {e}. Please ensure it is valid JSON.\033[0m")
                self.prompt_data = {}

        # 检查输入路径是目录还是 tar 文件
        if self.root.is_dir():
            if self.use_tar: #  如果 use_tar 为 True，则查找 tar 文件
                self.is_tar_input = True #  设置为 tar 输入模式
                self.tar_paths = [] #  修改：初始化为空列表，用于存储找到的 tar 文件路径
                index_counter = 0
                # 修改：使用 dirwalk 递归查找所有 tar 文件
                for tar_path in dirwalk(self.root):
                    if tar_path.suffix == '.tar':
                        self.tar_paths.append(tar_path)
                for tar_path in self.tar_paths:
                    meta_path = tar_path.with_suffix(".json") # 假设同名 json 文件
                    try:
                        with open(meta_path, 'r') as f:
                            self.tar_file_metas[tar_path] = json_lib.load(f)
                    except FileNotFoundError:
                        print(f"\033[33mWarning: Meta JSON file not found: {meta_path} for {tar_path}\033[0m")
                        continue # 如果 meta json 不存在，则跳过此 tar 文件
                    except json.JSONDecodeError:
                        print(f"\033[31mError decoding JSON in {meta_path} for {tar_path}. Please ensure it is valid JSON.\033[0m")
                        continue

                    if not self.tar_file_metas[tar_path] or 'files' not in self.tar_file_metas[tar_path]:
                        print(f"\033[33mWarning: Invalid meta JSON format in {meta_path} for {tar_path}. 'files' key is missing or empty.\033[0m")
                        continue

                    for filename_in_tar, file_info in self.tar_file_metas[tar_path]['files'].items():
                        if is_img(Path(filename_in_tar)): # 仅处理图像文件
                            self.paths.append(filename_in_tar) #  存储 tar 内的文件名
                            self.tar_index_map[index_counter] = (tar_path, filename_in_tar, file_info['offset'], file_info['size']) # 存储 tar 文件路径，文件名和偏移量/大小
                            index_counter += 1
                if self.is_tar_input and self.tar_paths: #  只有当找到 tar 文件时才打开
                    pass #  不再在 __init__ 中打开 tar 文件，而是在需要时打开
            else: #  如果 use_tar 为 False，则查找目录下的图片
                self.is_tar_input = False
                for artist_folder in self.root.iterdir():
                    if artist_folder.is_dir():
                        self.paths.extend(sorted(list(dirwalk(artist_folder, is_img))))
        elif self.root.suffix == '.tar': #  保持对直接输入 tar 文件的兼容
            self.is_tar_input = True
            self.tar_paths = [self.root] #  处理单个 tar 文件路径
            index_counter = 0
            for tar_path in self.tar_paths:
                meta_path = tar_path.with_suffix(".json") # 假设同名 json 文件
                try:
                    with open(meta_path, 'r') as f:
                        self.tar_file_metas[tar_path] = json_lib.load(f)
                except FileNotFoundError:
                    print(f"\033[33mWarning: Meta JSON file not found: {meta_path} for {tar_path}\033[0m")
                    continue # 如果 meta json 不存在，则跳过此 tar 文件
                except json.JSONDecodeError:
                    print(f"\033[31mError decoding JSON in {meta_path} for {tar_path}. Please ensure it is valid JSON.\033[0m")
                    continue

                if not self.tar_file_metas[tar_path] or 'files' not in self.tar_file_metas[tar_path]:
                    print(f"\033[33mWarning: Invalid meta JSON format in {meta_path} for {tar_path}. 'files' key is missing or empty.\033[0m")
                    continue

                for filename_in_tar, file_info in self.tar_file_metas[tar_path]['files'].items():
                    if is_img(Path(filename_in_tar)): # 仅处理图像文件
                        self.paths.append(filename_in_tar) #  存储 tar 内的文件名
                        self.tar_index_map[index_counter] = (tar_path, filename_in_tar, file_info['offset'], file_info['size']) # 存储 tar 文件路径，文件名和偏移量/大小
                        index_counter += 1
            if self.is_tar_input:
                pass #  不再在 __init__ 中打开 tar 文件，而是在需要时打开
        else:
            raise ValueError(f"Unsupported input root: {root}. Must be a directory or a .tar file.")

        print(f"Input root: {self.root}") # 打印输入的根路径
        print(f"Is tar input: {self.is_tar_input}") # 打印是否为 tar 文件输入

        self.dtype = dtype
        self.raw_res = []

        remove_paths = []
        for p_index in tqdm(
            range(len(self.paths)),
            desc="Loading image sizes",
            leave=False,
            ascii=True,
        ):
            try:
                if self.is_tar_input:
                    tar_path, filename_in_tar, offset, size = self.tar_index_map[p_index]
                    with tarfile.open(tar_path, 'r') as tar_file_handle:
                        img = self._load_entry_from_tar(tar_file_handle, offset, size) # 使用修改后的 _load_entry_from_tar 函数
                else:
                    img = load_entry(self.root / self.paths[p_index])
                h, w = Image.fromarray(img).size #  使用 Image.fromarray 获取 PIL Image 对象
                self.raw_res.append((h, w))
            except Exception as e:
                print(f"\033[33mSkipped: error processing {self.paths[p_index]}: {e}\033[0m")
                remove_paths.append(p_index) #  存储索引而不是路径

        remove_indices = set(remove_paths) #  使用索引集合
        self.paths = [self.paths[i] for i in range(len(self.paths)) if i not in remove_indices] #  根据索引过滤 paths
        self.raw_res = [self.raw_res[i] for i in range(len(self.raw_res)) if i not in remove_indices] #  根据索引过滤 raw_res
        if self.is_tar_input:
            new_tar_index_map = {}
            valid_index = 0
            for original_index, value in self.tar_index_map.items():
                if original_index not in remove_indices:
                    new_tar_index_map[valid_index] = value
                    valid_index += 1
            self.tar_index_map = new_tar_index_map # 更新 tar_index_map
        self.length = len(self.raw_res)
        print(f"Loaded {self.length} image sizes")
        
        self.fit_bucket_func = self.fit_bucket
        if no_upscale:
            self.fit_bucket_func = self.fit_bucket_no_upscale

        self.target_area = 1024 * 1024
        self.max_size, self.min_size, self.divisible = 4096, 256, 64
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
            diff = np.abs(np.log(self.bucket_ratios) - np.log(img_ratio))
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

    def _load_entry_from_tar(self, tar_file_handle, offset, size):
        tar_file_obj = tar_file_handle.fileobj  # 获取底层文件对象
        tar_file_obj.seek(offset)  # 定位到偏移量
        image_data = tar_file_obj.read(size)  # 读取指定大小的数据
        fileobj = io.BytesIO(image_data)  # 使用 BytesIO 包装字节数据
        _img = Image.open(fileobj)
        if _img.mode == "RGB":
            img = np.array(_img)
        elif _img.mode == "RGBA":
            baimg = Image.new('RGB', _img.size, (255, 255, 255))
            baimg.paste(_img, (0, 0), _img)
            img = np.array(baimg)
        else:
            img = np.array(_img.convert("RGB"))
        return img

    # 在 __getitem__ 方法中修改返回值
    def __getitem__(self, index) -> Entry:
        try:
            if self.is_tar_input:
                tar_path, filename_in_tar, offset, size = self.tar_index_map[index]
                with tarfile.open(tar_path, 'r') as tar_file_handle:
                    img = self._load_entry_from_tar(tar_file_handle, offset, size) # 使用修改后的 _load_entry_from_tar 函数
                image_path_str = filename_in_tar #  tar 文件中使用文件名作为 key
                full_image_path = Path(tar_path) / filename_in_tar #  为了 extras 里的 path 信息，需要构造一个路径，但实际上并不存在于文件系统
            else:
                img = load_entry(self.root / self.paths[index])
                image_path_str = str(self.paths[index]) #  目录输入使用相对路径
                full_image_path = self.root / self.paths[index] #  完整的路径

            original_size = img.shape[:2]
            img, dhdw = self.fit_bucket_func(index, img)
            img = self.tr(img).to(self.dtype)
            sha1 = get_sha1(full_image_path) #  sha1 计算使用构造的完整路径

            # 从 prompt_data 中获取 prompt 和 extra 信息
            if image_path_str in self.prompt_data:
                entry_data = self.prompt_data[image_path_str]
                prompt_general = entry_data.get('tag_string_general', '')
                extras = entry_data.copy()
            else:
                prompt_general = ""
                extras = {}
                print(f"\033[33mWarning: No metadata found for {self.paths[index]} in metadata JSON.\033[0m")


            extras.update({
                "path": str(full_image_path), #  使用构造的完整路径
                "train_caption": prompt_general,
                "sha1": sha1,
                "original_size": original_size,
                "dhdw": dhdw
            })

            return Entry(is_latent=False, pixel=img, prompt=prompt_general, extras=extras)
        except Exception as e:
            print(f"\033[31mError processing {self.paths[index]}: {e}\033[0m")
            return None

    def __len__(self):
        return len(self.paths)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="root directory or tar file of images" #  修改 help 描述
    )
    parser.add_argument("--output", "-o", type=str, required=True, help="output file")
    parser.add_argument("--no-upscale", "-nu", action="store_true", help="do not upscale images")
    parser.add_argument("--dtype", "-d", type=str, default="bfloat16", help="data type")
    parser.add_argument("--num_workers", "-n", type=int, default=6, help="number of dataloader workers")
    parser.add_argument("--metadata_json_path", "-metadata", type=str, default=None, help="path to metadata json file")
    parser.add_argument("--use_tar", "-ut", action="store_true", help="use tar files in the input directory") #  新增 use_tar 选项
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    root = args.input
    opt = Path(args.output)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    num_workers = args.num_workers
    metadata_json_path = args.metadata_json_path

    vae_path = "stabilityai/sdxl-vae"
    vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=dtype)
    vae.requires_grad_(False)
    vae.eval().cuda()

    dataset = LatentEncodingDataset(root, dtype=dtype, no_upscale=args.no_upscale, metadata_json_path=metadata_json_path, use_tar=args.use_tar)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    opt.mkdir(exist_ok=True, parents=True)
    assert opt.is_dir(), f"{opt} is not a directory"

    cache_filename = "cache.h5"
    dataset_mapping = {}

    h5_file_list = []
    current_file_index = 0
    max_file_size = 40 * 1024 * 1024 * 1024

    def create_new_h5_file(opt, current_file_index, h5_file_list):
        new_filename = f"cache_part_{current_file_index:03d}.h5"
        new_h5_cache_file = opt / new_filename
        h5_file_list.append(new_h5_cache_file)
        print(f"Creating new cache file: {new_h5_cache_file}")
        new_f = h5.File(new_h5_cache_file, "w", libver="latest")
        return new_f, new_h5_cache_file

    f, h5_cache_file = create_new_h5_file(opt, current_file_index, h5_file_list)

    with torch.no_grad():
        for entry in tqdm(dataloader):
            if entry is None:
                continue

            sha1 = entry.extras['sha1']
            w, h = entry.extras['original_size']
            
            dataset_mapping[sha1] = {
                "train_use": True,
                "train_caption": entry.extras['tag_string_general'],
                "file_path": entry.extras['path'],
                "train_width": w,
                "train_height": h,
                "extra": entry.extras
            }
            
            if f"{sha1}.latents" in f:
                print(f"\033[33mWarning: {entry.extras['path']} is already cached. Skipping... \033[0m")
                continue

            img = entry.pixel.unsqueeze(0).cuda()
            latent = vae.encode(img, return_dict=False)[0]
            latent.deterministic = True
            latent = latent.sample()[0]
            d = f.create_dataset(
                f"{sha1}.latents",
                data=latent.float().cpu().numpy(),
                compression="gzip",
            )
            d.attrs["scale"] = False
            d.attrs["dhdw"] = entry.extras['dhdw']

            if f.id.get_filesize() > max_file_size:
                f.close()
                current_file_index += 1
                f, h5_cache_file = create_new_h5_file(opt, current_file_index, h5_file_list)

    with open(opt / "dataset.json", "w") as f:
        json.dump(dataset_mapping, f, indent=4)