import sys
import os
import multiprocessing as mp
import numpy as np
from utilities.util import readNifti
import cc3d
import scipy.ndimage as ndi
import datetime

#TODO Many things only exists in the region dict, like patch overlap, average patch size - get them into metadatajso
#TODO Output is a dict of dicts, save as .pickledump in a specific directory maybe?
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
    #TODO Check that only patches in :overlap, :overlap, :overlap are counted/ patches which are also outside are -1
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
    diff = datetime.datetime.now() - beg
    proc_time = diff.microseconds
    count += max_v
    print(f"item {item} \t {count} \t {result_sum} \t {result_counter} || {count} {proc_time}",end="\r",flush=True)
    return (count, proc_time)

def counting(settings):
    global general_size_dict, result_sum, result_counter, total_time, path_patches, overlap

    path_patches = settings["paths"]["input_count_path"]# "/home/ramial-maskari/Documents/Segmentation/output/patches/prediction_Timing test2019-11-14 15:45:51.064475/"

    print(f"Patch path : {path_patches}")
    print(f"Number of processors : {mp.cpu_count()}")

    patch_list =[[i] for i in os.listdir(path_patches) if ".json" not in i]

    result_sum = 0
    result_counter = 0
    total_time = 0
    overlap = 5# region["partitioning"]["patch_overlap"]

    manager = mp.Manager()
    general_size_dict = manager.dict()


    pool = mp.Pool(mp.cpu_count())
    beg = datetime.datetime.now()

    for x in patch_list:
        pool.apply_async(process_item, args=(x), callback=sum_results)

    pool.close()
    pool.join()

    pool.close()
    print(f"\n{result_sum}")
    diff = datetime.datetime.now() - beg
    print(f"Multiprocessing took {diff}")
    final_dict = copy_dict(general_size_dict)
