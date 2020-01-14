import os

import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset, DataLoader
from utilities.util import find_divs, get_patch_data3d
"""
- Load a single item, split it into the wanted 
Pro:
    faster access to data (already in ram)
    no awkwardly large getitem method
    no multilist (dict of list of lists) as item but a simple dict
Cons:
    needs more ram 
    longer setup time (init should load stuff into RAM)
"""
class PredictionDataset(Dataset):
    #TODO Needs transformation, rotation, splits?
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings

        # Get paths
        nii_path = settings["paths"]["input_raw_path"]

        # create list
        nii_list = []

        mb_size = int(self.settings["dataloader"]["block_size"])

        # Load data
        for item in split:
            item_nii_path   = os.path.join(nii_path, item)

            image       = torch.tensor(np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1).astype(np.int64)).float()

            if transform:
                image       = transform(image)

            if norm:
                image       = norm(image)

            if image.shape[0] > mb_size:
                vdivs = find_divs(self.settings, image)
                offset_value = int(self.settings["preprocessing"]["padding"])
                offset_volume = (offset_value, offset_value, offset_value)
                offset_segmentation = (0,0,0)

                image_list = [x for x in get_patch_data3d(image, divs=vdivs, offset=offset_volume).unsqueeze(1)]

                nii_list = nii_list + image_list
            else:
                nii_list.append(image.unsqueeze(1))


        self.item_list = nii_list

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        return self.item_list[idx]
    
    def save_item(self, idx, item):
        print(f"idx {idx}\titem {item.shape}")
        #TODO Add batchsize to deduct list item indices

