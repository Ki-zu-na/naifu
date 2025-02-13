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
        """
        For tar-based image storage, we do not need to scan a directory.
        The 'root_path' can be a dummy value.
        Required kwargs:
          - tar_dirs: a list (or single value) of directories containing tar files.
          - metadata_json: path to the JSON file containing additional image metadata.
        """
        super().__init__(root_path, *args, **kwargs)

        # Retrieve tar_dirs and metadata_json from kwargs
        self.tar_dirs = self.kwargs.get("tar_dirs", [])
        if not self.tar_dirs:
            raise ValueError("tar_dirs parameter must be provided for TarImageStore.")
        self.metadata_json_path = self.kwargs.get("metadata_json", None) # 从 kwargs 中获取 metadata_json_path
        # if self.metadata_json is None: # TarImageStore 自己判断 metadata_json_path 是否为空
        #     raise ValueError("metadata_json parameter must be provided for TarImageStore.") # 如果 metadata_json_path 为空也不强制报错，允许不使用 metadata

        # Global metadata (e.g. tag/caption等信息)
        self.json_data = {}
        if self.metadata_json_path: # 只有当 metadata_json_path 存在时才加载和过滤
            try:
                with open(self.metadata_json_path, "r") as f:
                    self.json_data = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load metadata JSON: {e}")

        # 初始化集合
        self.tar_index = {}       # mapping: member.name -> (tar_path, TarInfo, file_info)
        self.filename_to_path = {}

        # Build tar index from tar_dirs, considering each tar file's associated JSON metadata.
        self._build_tar_index()

        # 过滤掉不在全局 metadata_json 中的条目 (只有当 metadata_json_path 存在时才进行过滤)
        if self.metadata_json_path and self.json_data:
            self.paths = [Path(name) for name in self.json_data.keys() if name in self.tar_index]
            print(f"TarImageStore filtered dataset to {len(self.paths)} entries based on metadata.")
        else: # 如果没有 metadata_json_path 或者 json_data 为空，则不过滤，使用所有 tar 文件中的条目
            self.paths = [Path(name) for name in self.tar_index if Path(name).is_file()] # 确保只包含文件路径
            print(f"TarImageStore loaded all {len(self.paths)} entries from tar files without metadata filtering.")

        self.length = len(self.paths)
        logger.debug(f"Filtered to {self.length} valid entries in TarImageStore.")

        # Set transform (reuse IMAGE_TRANSFORMS)
        self.transforms = IMAGE_TRANSFORMS

    def _build_tar_index(self):
        # 如果 tar_dirs 是一个字符串或 Path 对象，则转换为列表
        if isinstance(self.tar_dirs, (str, Path)):
            self.tar_dirs = [self.tar_dirs]

        # 用于存储每个 tar 文件对应的 JSON 元信息
        self.tar_json_data = {}
        for tar_dir in self.tar_dirs:
            tar_dir = Path(tar_dir)
            for tar_path in tar_dir.glob("**/*.tar"):
                print(f"Processing tar file: {tar_path}")
                # 查找同目录下与 tar 文件同名的 JSON 文件
                json_file = tar_path.with_suffix(".json")
                tar_file_metadata = {}
                if json_file.exists():
                    try:
                        with open(json_file, "r") as jf:
                            tar_file_metadata = json.load(jf)
                            self.tar_json_data[tar_path] = tar_file_metadata
                    except Exception as e:
                        logger.warning(f"Failed to load JSON for {tar_path}: {e}")
                else:
                    logger.warning(f"No JSON metadata found for {tar_path}")

                try:
                    with tarfile.open(tar_path, "r") as tf:
                        for member in tf.getmembers():
                            if member.isfile():
                                member_name = member.name
                                # 如果当前 tar 文件加载了元数据，并且包含 "files" 字段，则检查当前 member 是否在其中
                                if tar_file_metadata and "files" in tar_file_metadata:
                                    if member_name not in tar_file_metadata["files"]:
                                        continue  # 如果不在 JSON 描述中，则略过此成员
                                    file_info = tar_file_metadata["files"].get(member_name, {})
                                else:
                                    file_info = {}

                                if member_name not in self.tar_index:
                                    self.tar_index[member_name] = (tar_path, member, file_info)
                                    #print(f"    Added to index: {member_name}")
                                else:
                                    logger.warning(f"Duplicate entry for {member_name} found in {tar_path}, skipping.")

                                # 构建从文件名到成员完整路径的映射（仅当全局 metadata_json 中存在该文件时才添加）
                                filename = os.path.basename(member_name)
                                if filename in self.json_data:
                                    if filename not in self.filename_to_path:
                                        self.filename_to_path[filename] = member_name
                                        #print(f"filename_to_path added {filename} to {member_name}")
                                    else:
                                        logger.warning(f"Duplicate filename {filename} found. Keeping the first entry.")
                except Exception as e:
                    logger.error(f"Error processing tar file {tar_path}: {e}")
        print(f"Tar index built. Entries: {len(self.tar_index)}")
        print(f"filename_to_path built, Entries: {len(self.filename_to_path)}")

    def get_raw_entry(self, index) -> tuple[bool, torch.Tensor, str, tuple[int, int], tuple[int, int], dict]:
        # Get filename and metadata
        filename = str(self.paths[index])
        metadata = self.json_data.get(filename, {})

        # Retrieve tar info from index
        tar_info_pair = self.tar_index.get(filename, None)
        if tar_info_pair is None:
            raise FileNotFoundError(f"{filename} not found in any tar package")
        tar_path, tar_info = tar_info_pair

        # Open tar file and extract image data
        try:
            with tarfile.open(tar_path, "r") as tf:
                file_obj = tf.extractfile(tar_info)
                if file_obj is None:
                    raise FileNotFoundError(f"Couldn't extract {filename} from {tar_path}")
                img_data = file_obj.read()
        except Exception as e:
            raise RuntimeError(f"Error reading {filename} from {tar_path}: {e}")

        # Process image using PIL
        try:
            _img = Image.open(io.BytesIO(img_data))
            if _img.mode == "RGB":
                img = np.array(_img)
            elif _img.mode == "RGBA":
                baimg = Image.new("RGB", _img.size, (255, 255, 255))
                baimg.paste(_img, (0, 0), _img)
                img = np.array(baimg)
            else:
                img = np.array(_img.convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Error processing image {filename}: {e}")

        # Transform image to tensor
        img_tensor = self.transforms(img)
        h, w = img_tensor.shape[-2:]

        # Get prompt and extras from metadata
        prompt = metadata.get("tag_string_general", "")
        extras = metadata.copy()

        return False, img_tensor, prompt, (h, w), (0, 0), extras

class CombinedStore(StoreBase):
    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, *args, **kwargs)
        self._combined_paths = None  # 用于存储手动设置的 paths

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
            # Pop metadata_json from kwargs to avoid passing it twice
            tar_kwargs = kwargs.copy()
            tar_kwargs.pop("metadata_json", None)
            # 将 metadata_json_path 传递给 TarImageStore，让它自己处理过滤
            self.tar_store = TarImageStore(root_path, *args, metadata_json=self.metadata_json_path, **tar_kwargs)
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

        # 设置 dan_probability，可以从 kwargs 中获取或使用默认值 (为 process_batch_fn 做准备)
        dan_probability = kwargs.get('dan_probability', 0.7)

        # 创建一个偏函数，固定 dan_probability 参数
        self.process_entry = partial(shuffle_prompts_dan_native_style, dan_probability=dan_probability)

    @property
    def paths(self):
        # 如果已经手动设置过，则直接返回，否则聚合子 store 的 paths
        if self._combined_paths is not None:
            return self._combined_paths
        all_paths = []
        if self.latent_store is not None:
            all_paths += self.latent_store.paths
        if self.tar_store is not None:
            all_paths += self.tar_store.paths
        if self.directory_store is not None:
            all_paths += self.directory_store.paths
        return all_paths

    @paths.setter
    def paths(self, value):
        # 将赋值操作存储到 _combined_paths 中
        self._combined_paths = value

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