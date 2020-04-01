import sys
sys.path.append("/home/ramial-maskari/Documents/Pytorch Network/utilities")
import os
import numpy as np
from util import read_nifti, write_nifti, progress_bar
import cc3d
from scipy.ndimage.morphology import binary_erosion, binary_dilation

import pyqtgraph as pg

"""Idea: multiple iterations of cc3d and erosion on an array
         to filter out any big connected components (use median
         size of cells maybe?)
"""

def _get_bb(patch):
    """Get the bounding box for a cell in a binary matrix
    """
    a = np.where(patch > 0)
    bb = ((np.amin(a[0]), np.amin(a[1]), np.amin(a[2])), (np.amax(a[0]), np.amax(a[1]), np.amax(a[2])))
    return bb

def find_center(raw, patch):
    """Isolate the single cc in labels, cut out in raw, set the brightest point as center point
    """
    labels = cc3d.connected_components(patch) #XXX
    labels = patch
    result = np.zeros_like(raw) 
    max_l = np.amax(labels)
    for i in range(1, max_l):
        sub_label = np.copy(labels)
        sub_label[sub_label != i] = 0
        if (np.count_nonzero(sub_label > 0)) > 1:
            bb = _get_bb(sub_label)
            sub_label = sub_label[bb[0][0]:bb[1][0]+1,bb[0][1]:bb[1][1]+1,bb[0][2]:bb[1][2]+1]
            sub_raw = raw[bb[0][0]:bb[1][0]+1,bb[0][1]:bb[1][1]+1,bb[0][2]:bb[1][2]+1]
            sub_label[sub_label > 0] = 1
            sub_raw *= sub_label
            center_value = np.amax(sub_raw)
            center_coords = np.where(sub_raw == center_value)
            result[bb[0][0] + center_coords[0], bb[0][1] + center_coords[1], bb[0][2] + center_coords[2]] = i
        else:
            result += sub_label
    return result

def dilate(label, label_nr):
    label[label == label_nr] = 1
    label = binary_dilation(label)
    label[label == 1] = label_nr
    return label

def multi_cc3d(patch):
    """Get connected components, erode, get connected components again, see if there are more connected components afterwards
    """
    patch_c = np.copy(patch)
    labels = cc3d.connected_components(patch)
    max_labels = int(np.amax(labels))
    taken_l = max_labels
    for i in range(1, max_labels + 1):
        labels_c = np.copy(labels)
        labels_c[labels_c != i] = 0
        labels_c = labels_c / i
        sub_labels = binary_erosion(labels_c)
        sub_labels = cc3d.connected_components(sub_labels)
        print(f"Components {i} - {np.amax(sub_labels)}",end="\r",flush=True)
        if np.amax(sub_labels) > 1:
            # sub_labels[sub_labels > 1] = i
            labels[labels == i] = 0
            for sub_cc in range(1, np.amax(sub_labels)+1):
                if np.sum(sub_labels[sub_labels == sub_cc]) / sub_cc < 7:
                    sub_labels[sub_labels == sub_cc] = 0
            # TODO must binarize first and then back to the other number
            # TODO make sure every item is a unique number -> use taken_l
            # TODO always taken_l += 1
            # labels += binary_dilation(sub_labels) # XXX
            # copy sub labels into original label file
            # maybe go deeper?
        # TODO labels may overlap so every sub_label should be a unique number
        # TODO Add dilation
        # TODO Add erosion of un-cc3d labels
        # TODO Return and save instance labels instead of binary segmentation
        # Idea for another time: crop out the subpatch containing the cc
        # non_z = np.nonzero(labels_c)
        # coord_list = list(zip(non_z[0], non_z[1], non_z[2]))
    # labels[labels > 1] = 1 # XXX
    final_labels = cc3d.connected_components(labels)
    print(f"{max_labels} => {np.amax(final_labels)}") 
    return labels

def refine_gt(path_in, path_out):
    for item in os.listdir(path_in):
        print(f"Working on {item}...")
        # Loading data
        patch_path_in = os.path.join(path_in, item)
        patch_path_raw = os.path.join(path_raw, item)
        patch_raw = read_nifti(patch_path_raw)
        patch_gt = read_nifti(patch_path_in)

        # Processing data
        print(f"Multicc3d on {item}")
        patch_refined = multi_cc3d(patch_gt)
        print(f"Find centers on {item}")
        patch_center = find_center(patch_raw, patch_refined)

        # Writing data
        patch_path_out = os.path.join(path_out, item)
        patch_path_center = os.path.join(path_out, item.replace(".nii","_center.nii"))
        write_nifti(patch_path_out, patch_refined)
        write_nifti(patch_path_center, patch_center)

path_in = "/home/ramial-maskari/Documents/SHANEL Project/Segmentation/input/gt final/"
path_raw = "/home/ramial-maskari/Documents/SHANEL Project/Segmentation/input/raw/"
path_out = "/home/ramial-maskari/Documents/SHANEL Project/Segmentation/input/gt refined/"
refine_gt(path_in, path_out)
