import sys
import os
import multiprocessing as mp
import numpy as np
from utilities.util import readNifti, progress_bar
import cc3d
import scipy.ndimage as ndi
import datetime
import dill

#TODO Many things only exists in the region dict, like patch overlap, average patch size - get them into metadatajso
#TODO COMMMENTS XXX XXX XXX


def get_size_dict(arr, cutoff=-1):
    print("Getting Size dict")
    size_dict = {}
    max_v = np.amax(arr)
    for i in range(max_v + 1): 
        if i == 0:
            size = np.sum(arr[arr == i]) / 1 
        else:
            size = np.sum(arr[arr == i]) / i 

        if cutoff >= 0:
            if size > cutoff:
                size_dict[i] = size
        else:
            size_dict[i] = size
    return size_dict

def copy_dict(input_dict):
    res_dict = {}
    for key in input_dict:
        res_dict[key] = input_dict[key]
    return res_dict

def sum_results(result):
    global result_sum, result_counter, total_time
    res_s = result[0]
    res_time = result[1]
    result_sum += res_s
    result_counter += 1
    total_time += res_time

def process_item(item):
    global general_size_dict, path_patches, overlap
    count = 0
    beg = datetime.datetime.now()
    direct_path = os.path.join(path_patches, item)
    patch = readNifti(direct_path).astype(np.uint8)
    patch = patch[:-overlap,:-overlap, :-overlap]
    labels_out = cc3d.connected_components(patch)

    size_dict = {}
    max_v = int(np.amax(labels_out))
    for i in range(max_v + 1): 
        if i == 0:
            size = np.sum(labels_out[labels_out == i]) / 1 
        else:
            size = np.sum(labels_out[labels_out == i]) / i 
        size_dict[i] = size

    general_size_dict[f"{item}"] = size_dict
    progress_bar(len(general_size_dict.keys()), total_length, "Counting cells ")
    diff = datetime.datetime.now() - beg
    proc_time = diff.microseconds
    count += max_v
    return (count, proc_time)

def counting(settings):
    global general_size_dict, result_sum, result_counter, total_time, path_patches, overlap, total_length

    path_patches = settings["paths"]["input_count_path"]

    print(f"Patch path : {path_patches}")
    print(f"Number of processors : {mp.cpu_count()}")

    patch_list =[[i] for i in os.listdir(path_patches) if ".json" not in i]
    total_length = len(patch_list)

    result_sum = 0
    result_counter = 0
    total_time = 0
    overlap = settings["partitioning"]["overlap"]

    manager = mp.Manager()
    general_size_dict = manager.dict()


    pool = mp.Pool(mp.cpu_count())
    beg = datetime.datetime.now()

    for x in patch_list:
        pool.apply_async(process_item, args=(x), callback=sum_results)

    pool.close()
    pool.join()

    pool.close()
    print(f"\nFound {result_sum} in total.")
    diff = datetime.datetime.now() - beg
    print(f"Multiprocessing took {diff}")
    final_dict = copy_dict(general_size_dict)

    count_file_path = os.path.join(settings["paths"]["output_count_path"], settings["paths"]["model_name"] + "count_file.pickledump")
    with open(count_file_path, mode="w+b") as file:
        dill.dump(final_dict, file, protocol=4)
