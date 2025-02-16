import os
from pathlib import Path
import torch.distributed as dist
from glob import glob

def distributed_cache_tars(tar_files, output_dir, process_function):
    """
    Distribute the list of tar files across all ranks and process them in parallel.
    Each rank produces partial cache files saved in the output_dir.
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Each rank takes every world_size-th tar file
    tar_subset = tar_files[rank::world_size]

    cache_files = []
    for tar_file in tar_subset:
        cache_file = process_function(tar_file, output_dir)
        cache_files.append(cache_file)

    if dist.is_initialized():
        dist.barrier()

    # Rank 0 merges all partial caches into one final cache file.
    if rank == 0:
        merged_cache_path = os.path.join(output_dir, "final_cache.h5")
        merge_cache_files(output_dir, merged_cache_path)

    if dist.is_initialized():
        dist.barrier()

    return


def process_function(tar_file, output_dir):
    """
    Process a single tar file and store its cache into a partial cache file.
    Returns the path to the generated cache file.
    Note: Replace the dummy implementation with actual pre-caching logic.
    """
    import h5py
    from pathlib import Path

    cache_filename = f"cache_{Path(tar_file).stem}.h5"
    cache_path = os.path.join(output_dir, cache_filename)

    # Dummy processing: write a dummy dataset
    with h5py.File(cache_path, "w") as f:
        f.create_dataset("example", data=[1, 2, 3])
    
    return cache_path


def merge_cache_files(output_dir, merged_cache_path):
    """
    Merge all partial cache files in output_dir into final HDF5 cache files, each with a maximum size of 40GB.
    The output files will be named based on the merged_cache_path input, e.g., final_cache_0.h5, final_cache_1.h5, etc.
    """
    import h5py
    import os
    from glob import glob

    # 40GB 的字节数
    MAX_CACHE_SIZE = 40 * (1024 ** 3)

    partial_files = glob(os.path.join(output_dir, "cache_*.h5"))
    
    # 根据 merged_cache_path 提取基础文件名和扩展名
    base_name = os.path.splitext(os.path.basename(merged_cache_path))[0]
    ext = os.path.splitext(merged_cache_path)[1] if os.path.splitext(merged_cache_path)[1] else ".h5"
    
    current_file_index = 0
    current_merged_path = os.path.join(output_dir, f"{base_name}_{current_file_index}{ext}")
    merged_f = h5py.File(current_merged_path, "w")
    current_size = 0  # 当前文件累计的数据大小
    global_dataset_index = 0

    for part_file in partial_files:
        with h5py.File(part_file, "r") as part_f:
            for key in part_f.keys():
                data = part_f[key][:]
                data_size = data.nbytes

                # 如果当前文件大小加上新数据超过 40GB，则关闭当前文件，新建下一个文件
                if current_size + data_size > MAX_CACHE_SIZE:
                    merged_f.close()
                    current_file_index += 1
                    current_merged_path = os.path.join(output_dir, f"{base_name}_{current_file_index}{ext}")
                    merged_f = h5py.File(current_merged_path, "w")
                    current_size = 0

                merged_f.create_dataset(f"dataset_{global_dataset_index}", data=data)
                current_size += data_size
                global_dataset_index += 1

    merged_f.close()
    print(
        f"Merged {len(partial_files)} cache files into {current_file_index + 1} final cache files with a maximum of 40GB each"
    ) 