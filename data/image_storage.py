import os
import hashlib
import json
import h5py as h5
import numpy as np
import torch
import tarfile
import io
import tarfile
from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Generator, Optional  # type: ignore
from torchvision import transforms
from common.logging import logger

from data.processors import shuffle_prompts_dan_native_style
from functools import partial


json_lib = json
try:
    import rapidjson as json_lib
except ImportError:
    pass


image_suffix = set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"])

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def is_img(path: Path):
    return path.suffix in image_suffix


def sha1sum(txt):
    return hashlib.sha1(txt.encode()).hexdigest()


@dataclass
class Entry:
    is_latent: bool
    pixel: torch.Tensor
    prompt: str
    original_size: tuple[int, int]  # h, w
    cropped_size: Optional[tuple[int, int]]  # h, w
    dhdw: Optional[tuple[int, int]]  # dh, dw
    extras: dict = None
    # mask: torch.Tensor | None = None


def dirwalk(path: Path, cond: Optional[Callable] = None) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, cond)
        else:
            if isinstance(cond, Callable):
                if not cond(p):
                    continue
            yield p


class StoreBase(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        root_path,
        rank=0,
        dtype=torch.float16,
        process_batch_fn = "data.processors.shuffle_prompts_dan_native_style",
        **kwargs,
    ):
        self.rank = rank
        self.root_path = Path(root_path)
        self.dtype = dtype
        self.kwargs = kwargs
        self.process_batch_fn = process_batch_fn
            
        self.length = 0
        self.rand_list: list = []
        self.raw_res: list[tuple[int, int]] = []
        self.curr_res: list[tuple[int, int]] = []

        assert self.root_path.exists()

    def get_raw_entry(self, index) -> tuple[bool, np.ndarray, str, (int, int)]:
        raise NotImplementedError

    def fix_aspect_randomness(self, rng: np.random.Generator):
        raise NotImplementedError
    
    def crop(self, entry: Entry, index: int) -> Entry:
        return entry, 0, 0
    
    @torch.no_grad()
    def get_batch(self, indices: list[int]) -> Entry:
        entries = [self._get_entry(i) for i in indices]
        crop_pos = []
        pixels = []
        prompts = []
        original_sizes = []
        cropped_sizes = []
        extras = []



        for e, i in zip(entries, indices):
            e = self.process_batch(e)
            e, dh, dw = self.crop(e, i)
            pixels.append(e.pixel)
            original_size = torch.asarray(e.original_size)
            original_sizes.append(original_size)

            cropped_size = e.pixel.shape[-2:]
            cropped_size = (
                (cropped_size[0] * 8, cropped_size[1] * 8)
                if e.is_latent
                else cropped_size
            )
            cropped_size = torch.asarray(cropped_size)
            cropped_sizes.append(cropped_size)

            cropped_pos = (dh, dw)
            cropped_pos = (
                (cropped_pos[0] * 8, cropped_pos[1] * 8) if e.is_latent else cropped_pos
            )
            cropped_pos = (cropped_pos[0] + e.dhdw[0], cropped_pos[1] + e.dhdw[1])
            cropped_pos = torch.asarray(cropped_pos)
            crop_pos.append(cropped_pos)
            prompts.append(e.prompt)
            extras.append(e.extras)

        is_latent = entries[0].is_latent
        shape = entries[0].pixel.shape

        # Debugging: Print shapes before assertion
        print("Batch shapes before assertion:")
        for e in entries:
            print(f"  Shape: {e.pixel.shape}")

        for e in entries[1:]:
            assert (
                e.is_latent == is_latent
            ), f"Latent mismatch in batch"
            assert (
                e.pixel.shape == shape
            ), f"Shape mismatch in batch: {e.pixel.shape} != {shape}. First image shape: {shape},all_shape: {[item.pixel.shape for item in entries]}" 

        pixel = torch.stack(pixels, dim=0).contiguous()
        cropped_sizes = torch.stack(cropped_sizes)
        original_sizes = torch.stack(original_sizes)
        crop_pos = torch.stack(crop_pos)

        return {
            "prompts": prompts,
            "pixels": pixel,
            "is_latent": is_latent,
            "target_size_as_tuple": cropped_sizes,
            "original_size_as_tuple": original_sizes,
            "crop_coords_top_left": crop_pos,
            "extras": extras,
        }

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raise NotImplementedError

    def get_batch_extras(self, path):
        return {}

    def process_batch(self, inputs: Entry):
        if isinstance(self.process_batch_fn, str):
            self.process_batch_fn = get_class(self.process_batch_fn)
            
        return self.process_batch_fn(inputs)

    def _get_entry(self, index) -> Entry:
        is_latent, pixel, prompt, original_size, dhdw, extras = self.get_raw_entry(
            index
        )
        pixel = pixel.to(dtype=self.dtype)
        shape = pixel.shape
        if shape[-1] == 3 and shape[-1] < shape[0] and shape[-1] < shape[1]:
            pixel = pixel.permute(2, 0, 1)  # HWC -> CHW

        return Entry(is_latent, pixel, prompt, original_size, None, dhdw, extras)

    def repeat_entries(self, k, res, index=None):
        repeat_strategy = self.kwargs.get("repeat_strategy", None)
        if repeat_strategy is not None:
            assert index is not None
            index_new = index.copy()
            for i, ent in enumerate(index):
                for strategy, mult in repeat_strategy:
                    if strategy in str(ent):
                        k.extend([k[i]] * (mult - 1))
                        res.extend([res[i]] * (mult - 1))
                        index_new.extend([index_new[i]] * (mult - 1))
                        break
        else:
            index_new = index
        return k, res, index_new

class LatentStore(StoreBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prompt_mapping = next(dirwalk(self.root_path, lambda p: p.suffix == ".json"))
        prompt_mapping = json_lib.loads(Path(prompt_mapping).read_text())

        self.h5_paths = list(
            dirwalk(
                self.root_path,
                lambda p: p.suffix == ".h5" and "prompt_cache" not in p.stem,
            )
        )
        
        self.h5_keymap = {}
        self.h5_filehandles = {}
        self.paths = []
        self.keys = []
        progress = tqdm(
            total=len(prompt_mapping),
            desc=f"Loading latents",
            disable=self.rank != 0,
            leave=False,
            ascii=True,
        )

        has_h5_loc = "h5_path" in next(iter(prompt_mapping.values()))
        for idx, h5_path in enumerate(self.h5_paths):
            fs = h5.File(h5_path, "r", libver="latest")
            h5_name = h5_path.name
            
            for k in fs.keys():
                hashkey = k[:-8]  # ".latents"
                if hashkey not in prompt_mapping:
                    #logger.warning(f"Key {k} not found in prompt_mapping")
                    continue
                
                it = prompt_mapping[hashkey]
                if not it["train_use"] or (has_h5_loc and it["h5_path"] != h5_name):
                    continue
                
                height, width, fp = it["train_height"], it["train_width"], it["file_path"]
                self.paths.append(fp)
                self.keys.append(k)
                self.raw_res.append((height, width))
                self.h5_keymap[k] = (h5_path, it, (height, width))
                progress.update(1)
                
        progress.close()
        self.length = len(self.keys)
        self.scale_factor = 0.13025
        logger.debug(f"Loaded {self.length} latent codes from {self.root_path}")

        self.keys, self.raw_res, self.paths = self.repeat_entries(self.keys, self.raw_res, index=self.paths)
        new_length = len(self.keys)
        if new_length != self.length:
            self.length = new_length
            logger.debug(f"Using {self.length} entries after applied repeat strategy")

        # 设置 dan_probability，可以从 kwargs 中获取或使用默认值
        dan_probability = kwargs.get('dan_probability', 0.7)
        
        # 创建一个偏函数，固定 dan_probability 参数
        self.process_entry = partial(shuffle_prompts_dan_native_style, dan_probability=dan_probability)


    def setup_filehandles(self):
        self.h5_filehandles = {}
        for h5_path in self.h5_paths:
            self.h5_filehandles[h5_path] = h5.File(h5_path, "r", libver="latest")

    
    def get_raw_entry(self, index) -> tuple[bool, torch.tensor, str, tuple[int, int], tuple[int, int], dict]:
        if len(self.h5_filehandles) == 0:
            self.setup_filehandles()
            
        latent_key = self.keys[index]
        h5_path, entry, original_size = self.h5_keymap[latent_key]
        
        # 获取原始 prompt
        prompt = entry["train_caption"]

        latent = torch.asarray(self.h5_filehandles[h5_path][latent_key][:]).float()
        dhdw = self.h5_filehandles[h5_path][latent_key].attrs.get("dhdw", (0, 0))

        # if scaled, we need to unscale the latent (training process will scale it back)
        scaled = self.h5_filehandles[h5_path][latent_key].attrs.get("scale", True)
        if scaled:
            latent = 1.0 / self.scale_factor * latent

        # 获取额外信息
        extras = self.get_batch_extras(self.paths[index])
        
        # 添加必要的信息到 extras
        extras['train_caption_dan'] = entry.get('train_caption_dan', prompt)
        extras['train_caption_native'] = entry.get('train_caption_native', prompt)

        return True, latent, prompt, original_size, dhdw, extras


class DirectoryImageStore(StoreBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 获取配置参数
        self.label_ext = self.kwargs.get("label_ext", ".txt")  # 文本文件扩展名
        self.json_path = self.kwargs.get("json_path", None)  # json文件路径
        self.json_data = {}
        
        if self.json_path:
            with open(self.json_path, 'r') as f:
                for line in f:
                    entry = json_lib.loads(line)
                    img_path = entry['image_path']
                    # 复制所有字段到extras，除了image_path
                    self.json_data[img_path] = {k: v for k, v in entry.items() if k != 'image_path'}

        self.paths = list(dirwalk(self.root_path, is_img))
        self.length = len(self.paths)
        self.transforms = IMAGE_TRANSFORMS
        logger.debug(f"Found {self.length} images in {self.root_path}")

        remove_paths = []
        for p in tqdm(
            self.paths,
            desc="Loading image sizes",
            leave=False,
            ascii=True,
        ):
            try:
                w, h = Image.open(p).size
                self.raw_res.append((h, w))
            except Exception as e:
                print(f"\033[33mSkipped: error processing {p}: {e}\033[0m")
                remove_paths.append(p)

        remove_paths = set(remove_paths)
        self.paths = [p for p in self.paths if p not in remove_paths]
        self.length = len(self.raw_res)

        self.length = len(self.paths)
        self.prompts: list[str] = []
        for path in tqdm(
            self.paths,
            desc="Loading prompts",
            disable=self.rank != 0,
            leave=False,
            ascii=True,
        ):
            p = path.with_suffix(self.label_ext)
            try:
                with open(p, "r") as f:
                    self.prompts.append(f.read())
            except Exception as e:
                logger.warning(f"Skipped: error processing {p}: {e}")
                self.prompts.append("")
                
        self.prompts, self.raw_res, self.paths = self.repeat_entries(
            self.prompts, self.raw_res, index=self.paths
        )
        new_length = len(self.paths)
        if new_length != self.length:
            self.length = new_length
            logger.debug(f"Using {self.length} entries after applied repeat strategy")

    def get_raw_entry(self, index) -> tuple[bool, torch.tensor, str, tuple[int, int], tuple[int, int], dict]:
        p = self.paths[index]
        
        if self.jsonl_path:
            img_path = str(p)
            if img_path in self.json_data:
                data = self.json_data[img_path]
                prompt = data.get('tag_string_general', '')
                # 将所有json字段存入extras
                extras = data.copy()
            else:
                prompt = ''
                extras = {}
        else:
            # 从txt文件读取prompt
            try:
                with open(p.with_suffix(self.label_ext), "r") as f:
                    prompt = f.read()
                extras = {}
            except Exception as e:
                logger.warning(f"跳过: 处理{p}时出错: {e}")
                prompt = ""
                extras = {}

        # 处理图片
        _img = Image.open(p)
        if _img.mode == "RGB":
            img = np.array(_img)
        elif _img.mode == "RGBA":
            baimg = Image.new('RGB', _img.size, (255, 255, 255))
            baimg.paste(_img, (0, 0), _img)
            img = np.array(baimg)
        else:
            img = np.array(_img.convert("RGB"))

        img = self.transforms(img)
        h, w = img.shape[-2:]
        dhdw = (0, 0)

        return False, img, prompt, (h, w), dhdw, extras

class TarImageStore(StoreBase):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self.metadata_json_path = kwargs.get("metadata_json")  # 指定文件json路径
        self.prompt_data = {}
        if self.metadata_json_path:
            try:
                with open(self.metadata_json_path, 'r') as f:
                    self.prompt_data = json_lib.load(f)
            except FileNotFoundError:
                logger.warning(f"Prompt JSON file not found: {self.metadata_json_path}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON in {self.metadata_json_path}. Please ensure it is valid JSON.")
                self.prompt_data = {}

        self.tar_paths = list(dirwalk(self.root_path, lambda p: p.suffix == ".tar"))
        self.tar_file_metas = {} # 用于存储 tar 文件元数据 (offset 信息)
        self.paths = [] # 存储文件路径 (在 tar 中的文件名)
        self.raw_res = [] # 存储原始分辨率
        self.tar_index_map = {} # 索引到 (tar_path, filename_in_tar) 的映射

        index_counter = 0
        for tar_path in self.tar_paths:
            meta_path = tar_path.with_suffix(".json") # 假设同名 json 文件

            try:
                with open(meta_path, 'r') as f:
                    self.tar_file_metas[tar_path] = json_lib.load(f)
            except FileNotFoundError:
                logger.warning(f"Meta JSON file not found: {meta_path} for {tar_path}")
                continue # 如果 meta json 不存在，则跳过此 tar 文件
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON in {meta_path} for {tar_path}. Please ensure it is valid JSON.")
                continue

            if not self.tar_file_metas[tar_path] or 'files' not in self.tar_file_metas[tar_path]:
                logger.warning(f"Invalid meta JSON format in {meta_path} for {tar_path}. 'files' key is missing or empty.")
                continue

            for filename_in_tar, file_info in self.tar_file_metas[tar_path]['files'].items():
                if is_img(Path(filename_in_tar)): # 仅处理图像文件
                    self.paths.append(filename_in_tar)
                    try:
                        with tarfile.open(tar_path, 'r') as tar_file_handle: #  打开 tar 文件
                            tar_file_obj = tar_file_handle.fileobj  # 获取底层文件对象
                            tar_file_obj.seek(file_info['offset'])  # 定位到偏移量
                            image_data = tar_file_obj.read(file_info['size'])  # 读取指定大小的数据
                            fileobj = io.BytesIO(image_data) # 使用 BytesIO 包装字节数据
                            _img = Image.open(fileobj)
                            height, width = _img.size[1], _img.size[0]
                            self.raw_res.append((height, width))
                    except Exception as e:
                        logger.warning(f"无法从图像文件 {filename_in_tar} 中获取分辨率: {e}, 使用默认分辨率 (1024, 1024)")
                        self.raw_res.append((1024, 1024)) # 默认分辨率
                    self.tar_index_map[index_counter] = (tar_path, filename_in_tar)
                    index_counter += 1

        self.length = len(self.tar_index_map)
        logger.debug(f"Found {self.length} images in tar archives from {self.root_path}")

        self.paths, self.raw_res, _ = self.repeat_entries(self.paths, self.raw_res, index=list(range(self.length))) # index 这里用 range(length) 即可
        new_length = len(self.paths)
        if new_length != self.length:
            self.length = new_length
            logger.debug(f"Using {self.length} entries after applied repeat strategy")

        # 设置 dan_probability
        dan_probability = kwargs.get('dan_probability', 0.7)
        self.process_entry = partial(shuffle_prompts_dan_native_style, dan_probability=dan_probability)


    def get_raw_entry(self, index) -> tuple[bool, torch.tensor, str, tuple[int, int], tuple[int, int], dict]:
        tar_path, filename_in_tar = self.tar_index_map[index]
        tar_meta = self.tar_file_metas[tar_path]
        file_meta = tar_meta['files'][filename_in_tar]

        offset = file_meta['offset']
        size = file_meta['size']

        try:
            with tarfile.open(tar_path, 'r') as tar_file_handle: # 在这里打开 tar 文件
                with tar_file_handle.fileobj as tar_file_obj: #  使用 tar_file_handle.fileobj
                    tar_file_obj.seek(offset)
                    image_data = tar_file_obj.read(size)
                    fileobj = io.BytesIO(image_data)
                    _img = Image.open(fileobj)
                    if _img.mode == "RGB":
                        img = np.array(_img)
                    elif _img.mode == "RGBA":
                        baimg = Image.new('RGB', _img.size, (255, 255, 255))
                        baimg.paste(_img, (0, 0), _img)
                        img = np.array(baimg)
                    else:
                        img = np.array(_img.convert("RGB"))
        except Exception as e:
            logger.error(f"Error reading image {filename_in_tar} from {tar_path}: {e}")
            # 返回一个占位符图像或引发异常，根据您的错误处理策略
            # 这里为了保证dataset能继续运行，返回一个 None 图片和空 prompt
            placeholder_img = np.zeros((512, 512, 3), dtype=np.uint8) # 占位符图像
            img = placeholder_img
            prompt = ""
            original_size = (512, 512)
            dhdw = (0, 0)
            extras = {}
            return False, torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0, prompt, original_size, dhdw, extras # 返回处理后的占位符图像


        img = IMAGE_TRANSFORMS(img)
        h, w = img.shape[-2:]
        original_size = (h, w)
        assert self.raw_res[index] == (h,w), f"Shape mismatch in batch: {self.raw_res[index]} != {original_size}"
        dhdw = (0, 0)

        # 获取 prompt 和 extras
        prompt = ""
        extras = {}
        if filename_in_tar in self.prompt_data:
            entry_data = self.prompt_data[filename_in_tar]
            prompt = entry_data.get('tag_string_general', '') # 或者其他 prompt 字段
            extras = entry_data.copy()

        return False, img, prompt, original_size, dhdw, extras


class CombinedStore(StoreBase):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self._combined_paths = []  # 用于存储合并后的 paths
        self._combined_raw_res = [] # 用于存储合并后的 raw_res

        self.metadata_json_path = self.kwargs.get("metadata_json")
        self.tar_dirs = self.kwargs.get("tar_dirs", [])
        self.load_latent = self.kwargs.get("load_latent", False)  # 显式指定是否加载 latent
        self.load_tar = self.kwargs.get("load_tar", False)
        self.load_directory = self.kwargs.get("load_directory", False)
        self.metadata_json = {}

        self.latent_store = None
        self.tar_store = None
        self.directory_store = None

        current_index = 0

        # Initialize and load data for each store type
        if self.load_latent:
            self.latent_store = LatentStore(root_path, *args, **kwargs)
            self.latent_length = len(self.latent_store)
            self.latent_index_map = {
                i: (current_index + i, "latent") for i in range(self.latent_length)
            }
            current_index += self.latent_length

            self.latent_store.setup_filehandles() # 确保文件句柄被初始化

        if self.load_tar:
            self.tar_store = TarImageStore(root_path, *args, **kwargs)
            self.tar_length = len(self.tar_store)
            self.tar_index_map = {
                i: (current_index + i, "tar") for i in range(self.tar_length)
            }
            current_index += self.tar_length

        if self.load_directory:
            self.directory_store = DirectoryImageStore(root_path, *args, **kwargs)
            self.directory_length = len(self.directory_store)
            self.directory_index_map = {
                i: (current_index + i, "directory")
                for i in range(self.directory_length)
            }
            current_index += self.directory_length

        self.length = current_index  # Total length of the combined dataset

        # 合并所有子 Store 的 paths 和 raw_res
        if self.latent_store is not None:
            self._combined_paths.extend(self.latent_store.paths)
            self._combined_raw_res.extend(self.latent_store.raw_res)
        if self.tar_store is not None:
            self._combined_paths.extend(self.tar_store.paths)
            self._combined_raw_res.extend(self.tar_store.raw_res)
        if self.directory_store is not None:
            self._combined_paths.extend(self.directory_store.paths)
            self._combined_raw_res.extend(self.directory_store.raw_res)

        self.paths = self._combined_paths # 直接赋值，paths 不再是 property
        self.raw_res = self._combined_raw_res # 直接赋值，raw_res 也不再是 property

        # 设置 dan_probability，可以从 kwargs 中获取或使用默认值 (为 process_batch_fn 做准备)
        dan_probability = kwargs.get('dan_probability', 0.7)

        # 创建一个偏函数，固定 dan_probability 参数
        self.process_entry = partial(shuffle_prompts_dan_native_style, dan_probability=dan_probability)

    def get_raw_entry(self, index):
        # Determine which store to use based on the index
        if self.load_latent and index in self.latent_index_map:
            mapped_index, store_type = self.latent_index_map[index]
            is_latent, pixel, prompt, original_size, dhdw, extras = self.latent_store.get_raw_entry(
                mapped_index - (0 if store_type == "latent" else (self.latent_length if store_type == "tar" else self.latent_length + self.tar_length))
            )
        elif self.load_tar and index in self.tar_index_map:
            mapped_index, store_type = self.tar_index_map[index]
            is_latent, pixel, prompt, original_size, dhdw, extras = self.tar_store.get_raw_entry(
                mapped_index - (0 if store_type == "tar" else (self.latent_length if store_type == "latent" else self.tar_length + self.directory_length))
            )
        elif self.load_directory and index in self.directory_index_map:
            mapped_index, store_type = self.directory_index_map[index]
            is_latent, pixel, prompt, original_size, dhdw, extras = self.directory_store.get_raw_entry(
                mapped_index - (0 if store_type == "directory" else (self.tar_length if store_type == "tar" else self.directory_length))
            )
        else:
            raise IndexError(f"Index {index} out of range for CombinedStore")

        # Merge extras with metadata from metadata_json, if available
        if self.metadata_json:
            if store_type == "latent":
                filename = self.latent_store.paths[
                    mapped_index - (0 if store_type == "latent" else (self.latent_length if store_type == "tar" else self.latent_length + self.tar_length))
                ]
                metadata_key = filename
            elif store_type == "tar":
                filename = os.path.basename(str(self.tar_store.paths[
                    mapped_index - (0 if store_type == "tar" else (self.latent_length if store_type == "latent" else self.tar_length + self.directory_length))
                ]))
                metadata_key = filename
            else:
                filename = os.path.basename(str(self.directory_store.paths[
                    mapped_index - (0 if store_type == "directory" else (self.tar_length if store_type == "tar" else self.directory_length))
                ]))
                metadata_key = filename

            if metadata_key in self.metadata_json:
                extras.update(self.metadata_json[metadata_key])

        return is_latent, pixel, prompt, original_size, dhdw, extras

    def __len__(self):
        return self.length

    def setup_filehandles(self):  # 新增方法
        if self.latent_store:
            self.latent_store.setup_filehandles()
        if self.tar_store:
            self.tar_store.setup_filehandles()

    def get_batch_extras(self, path): # 这个函数现在可能用处不大，但为了兼容性，可以保留一个空的实现
        return {}