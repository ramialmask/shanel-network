import sys
sys.path.append("/home/ramial-maskari/Documents/Pytorch Network/utilities/")
import os
import numpy as np
from util import read_nifti, write_nifti, progress_bar
import cc3d
from scipy.ndimage.morphology import binary_erosion

"""Idea: multiple iterations of cc3d and erosion on an array
         to filter out any big connected components (use median
         size of cells maybe?)
"""

def multi_cc3d(patch):
    patch_c = np.copy(patch)
    labels = cc3d.connected_components(patch)
    max_labels = int(np.amax(labels))
    print(f"Found {max_labels} components")
    for i in range(1, max_labels + 1):
        labels_c = np.copy(labels)
        labels_c[labels_c != i] = 0
        labels_c = labels_c / i
        sub_labels = binary_erosion(labels_c)
        sub_labels = cc3d.connected_components(sub_labels)
        print(f"Components {i} - {np.amax(sub_labels)}",end="\r",flush=True)
        if np.amax(sub_labels) > 0:
            sub_labels[sub_labels > 1] = i
            labels[labels == i] = 0
            labels += sub_labels
            # copy sub labels into original label file
            # maybe go deeper?
        # Idea for another time: crop out the subpatch containing the cc
        # non_z = np.nonzero(labels_c)
        # coord_list = list(zip(non_z[0], non_z[1], non_z[2]))
    return labels

def refine_gt(path_in, path_out):
    for item in os.listdir(path_in)[:2]:
        print(f"Working on {item}...")
        patch_path_in = os.path.join(path_in, item)
        patch = read_nifti(patch_path_in)
        patch_refined = multi_cc3d(patch)

        patch_path_out = os.path.join(path_out, item)
        print(f"Writing to {patch_path_out}")
        write_nifti(patch_path_out,patch_refined)

path_in = "/home/ramial-maskari/Documents/Doris Segmentation/input/gt/"
path_out = "/home/ramial-maskari/Documents/Doris Segmentation/input/gt refined/"
refine_gt(path_in, path_out)
