import os
import hashlib
import json
import h5py as h5
import numpy as np
import torch
import tarfile
import io

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

        for e in entries[1:]:
            assert e.is_latent == is_latent
            assert (
                e.pixel.shape == shape
            ), f"{e.pixel.shape} != {shape} for the same batch"

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
        self.tar_dirs = kwargs.get("tar_dirs", [])
        self.metadata_json_path = self.kwargs.get("metadata_json")
        self.metadata_json = {}
        self.paths = []
        self.raw_res = []
        self.filename_to_path = {} # filename -> tar path
        self.tar_index = [] # 存储 tar 文件索引信息，每个元素是 (tar_path, index_in_tar)
        self.length = 0

        print(f"[TarImageStore] Initial tar_dirs: {self.tar_dirs}") # 调试日志：打印初始 tar_dirs

        # 确保 tar_dirs 是一个列表
        if isinstance(self.tar_dirs, str):
            self.tar_dirs = [self.tar_dirs]
        elif not isinstance(self.tar_dirs, list):
            raise TypeError("tar_dirs should be a string or a list of strings")

        print(f"[TarImageStore] Processed tar_dirs: {self.tar_dirs}") # 调试日志：打印处理后的 tar_dirs，确认是否为列表

        if self.metadata_json_path:
            try:
                with open(self.metadata_json_path, 'r', encoding='utf-8') as f:
                    self.metadata_json = json.load(f)
                print(f"[TarImageStore] Metadata loaded from {self.metadata_json_path}")
            except FileNotFoundError:
                print(f"[TarImageStore] Metadata file not found at {self.metadata_json_path}, proceeding without metadata filtering.")
            except json.JSONDecodeError as e:
                print(f"[TarImageStore] Error decoding metadata JSON at {self.metadata_json_path}: {e}. Proceeding without metadata filtering.")
                self.metadata_json = {} # 确保即使解析错误也初始化为空字典，避免后续代码出错

        if not self.tar_dirs:
            self.tar_dirs = [self.root_path]

        self._build_index_from_tar()
        if self.metadata_json:
            self._filter_by_metadata() # 添加根据 metadata 过滤的步骤
        else:
            print("[TarImageStore] No metadata JSON provided, skipping metadata filtering.")

        print(f"[TarImageStore] Total entries after filtering: {len(self.paths)}")
        if not self.paths:
            print("[TarImageStore] WARNING: No valid image paths found after filtering!")
        else:
            print(f"[TarImageStore] First 10 image paths: {self.paths[:10]}")


    def _build_index_from_tar(self):
        all_entries = []
        filename_to_path = {}
        for tar_dir in self.tar_dirs:
            tar_dir_path = Path(tar_dir)
            if not tar_dir_path.exists():
                print(f"[TarImageStore] Warning: Tar directory '{tar_dir_path}' does not exist.")
                continue # 如果目录不存在，跳过

            for tar_path in tar_dir_path.glob("*.tar"):
                print(f"Processing tar file: {tar_path}")
                with tarfile.open(tar_path, 'r') as tar:
                    members = tar.getmembers()
                    for i, member in enumerate(members):
                        if member.isfile() and Path(member.name).suffix.lower() in image_suffix:
                            filename = Path(member.name).name
                            all_entries.append((str(tar_path), i, filename)) # 存储 tar 文件路径, 索引和文件名
                            filename_to_path[filename] = str(tar_path) # 构建 filename 到 tar 路径的映射
        self.tar_index = all_entries
        self.filename_to_path = filename_to_path
        print(f"Tar index built. Entries: {len(self.tar_index)}")
        print(f"filename_to_path built, Entries: {len(self.filename_to_path)}")


    def _filter_by_metadata(self):
        valid_entries = []
        valid_paths = []
        valid_raw_res = []
        filtered_count = 0

        if not self.metadata_json:
            print("[TarImageStore] No metadata to filter with, skipping filtering.")
            return # 如果 metadata_json 为空，直接返回

        valid_ids_from_metadata = set(self.metadata_json.keys())
        print(f"[TarImageStore] Number of valid IDs from metadata: {len(valid_ids_from_metadata)}")

        for tar_path, index_in_tar, filename in self.tar_index:
            filename_without_ext = Path(filename).stem # 获取不带扩展名的文件名
            if filename_without_ext in valid_ids_from_metadata:
                valid_entries.append((tar_path, index_in_tar))
                valid_paths.append(filename) # 这里使用文件名，而不是完整路径，因为 paths 属性通常用于文件名
                # 需要从 tar 文件中读取图像以获取 raw_res，这里先放一个占位符，稍后实现
                try:
                    is_latent, pixel, prompt, original_size, dhdw, extras = self.get_raw_entry_from_tar_entry((tar_path, index_in_tar))
                    valid_raw_res.append(original_size)
                except Exception as e:
                    print(f"[TarImageStore] Error getting raw entry for {filename} in {tar_path}: {e}. Skipping resolution retrieval.")
                    valid_raw_res.append((512, 512)) # 默认分辨率，防止后续代码出错
            else:
                filtered_count += 1
                # print(f"[TarImageStore] Filtered out {filename} because it's not in metadata.") # 太多过滤信息，默认不打印

        self.tar_index = valid_entries
        self.paths = valid_paths
        self.raw_res = valid_raw_res
        self.length = len(self.paths)

        print(f"DEBUG - Filtered to {len(self.paths)} valid entries in TarImageStore.")
        if filtered_count > 0:
            print(f"[TarImageStore] Filtered dataset to {len(self.paths)} entries based on metadata. {filtered_count} entries removed.")
        else:
            print(f"[TarImageStore] No entries were filtered out based on metadata.")


    def get_raw_entry_from_tar_entry(self, tar_entry):
        tar_path, index_in_tar = tar_entry
        with tarfile.open(tar_path, 'r') as tar:
            member = tar.getmembers()[index_in_tar]
            file = tar.extractfile(member)
            _img = Image.open(io.BytesIO(file.read()))
            metadata = {}
            filename = Path(member.name).name

            if self.metadata_json and Path(filename).stem in self.metadata_json:
                 metadata = self.metadata_json[Path(filename).stem]

            if _img.mode != "RGB":
                _img = _img.convert("RGB")

            img = np.array(_img)
            h, w = img.shape[:2]
            prompt = metadata.get("tag_string_general", "")
            extras = metadata.copy()

            return False, torch.from_numpy(img), prompt, (h, w), (0, 0), extras


    def get_raw_entry(self, index):
        tar_path, index_in_tar = self.tar_index[index]
        return self.get_raw_entry_from_tar_entry((tar_path, index_in_tar))


    def __len__(self):
        return self.length


    def __getitem__(self, index):
        return self._get_entry(index)


    def setup_filehandles(self):
        pass # Tar files are opened and closed within get_raw_entry, no need to keep file handles open


    def get_batch_extras(self, path):
        filename = os.path.basename(path) # 提取文件名
        if self.metadata_json and filename in self.metadata_json:
            return {"metadata": self.metadata_json[filename]} # 返回整个 metadata 条目
        return {} # 默认返回空字典


    @property
    def image_paths(self): # 兼容性属性，返回 paths 的复制
        return list(self.paths)

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

    def get_batch_extras(self, path): # 这个函数现在可能用处不大，但为了兼容性，可以保留一个空的实现
        return {}