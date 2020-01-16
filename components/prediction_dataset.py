import os

import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset, DataLoader
from utilities.util import find_divs, get_patch_data3d, get_volume_from_patches3d, writeNifti
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
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings
        self.split = split
        # Get paths
        nii_path = settings["paths"]["input_raw_path"]

        # create list
        nii_list = []

        mb_size = int(self.settings["dataloader"]["block_size"])

        self.vdivs = -1
        self.offset_volume = -1
        # Load data
        for item in split:
            item_nii_path = os.path.join(nii_path, item)
            image = torch.tensor(np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1).astype(np.int64)).float()

            if transform:
                image = transform(image)

            if norm:
                image = norm(image)

            if image.shape[0] > mb_size:
                self.vdivs = find_divs(self.settings, image)
                offset_value = int(self.settings["preprocessing"]["padding"])
                self.offset_volume = (offset_value, offset_value, offset_value)

                image_list = [x for x in get_patch_data3d(image, divs=self.vdivs, offset=self.offset_volume).unsqueeze(1)]

                nii_list = nii_list + image_list
            else:
                nii_list.append(image.unsqueeze(1))


        self.item_list = nii_list

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        return self.item_list[idx]

    def save_item(self, idx, item):
        batch_size = int(self.settings["dataloader"]["batch_size"])
        output_path = self.settings["paths"]["output_seg_path"]
        if not self.vdivs == -1:
            # get volume from patch data 3d
            item = np.squeeze(item)
            direct_item = get_volume_from_patches3d(item, divs=self.vdivs)#, offset=self.offset_volume)
        else:
            # just save the volume 
            direct_item = item
          
        direct_item = direct_item.numpy().astype(np.int32)
        direct_item_name = self.split[idx]
        direct_output_path = os.path.join(output_path, direct_item_name)
        writeNifti(direct_output_path, direct_item)



