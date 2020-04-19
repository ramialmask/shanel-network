import os

import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset, DataLoader
from utilities.util import find_divs, get_patch_data3d
"""
- Load all datasets into one huge list
- iterate over this list as method of getitem
Pro:
    faster access to data (already in ram)
    no awkwardly large getitem method
    no multilist (dict of list of lists) as item but a simple dict
Cons:
    needs more ram 
    longer setup time (init should load stuff into RAM)
"""
class TrainingDatasetDiscriminator_3D(Dataset):
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings

        # Get paths
        nii_path = settings["paths"]["input_path"]

        # create list
        nii_list = []

        # Load data
        for item in split:
            item_nii_path   = os.path.join(nii_path, item)

            image       = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1).astype(np.float64)
            image_class = -1
            if "real" in item:
                image_class = 0
            elif "syn" in item:
                image_class = 1
            image_class = np.array(image_class).astype(np.float64)

            if transform:
                image       = transform(image)

            if norm:
                image       = norm(image)
            
            # Torchify
            image = torch.tensor(image).float()
            image_class = torch.tensor(image_class).float()

            nii_list.append((image.unsqueeze(0), image_class))


        self.item_list = nii_list

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        return {"volume":self.item_list[idx][0], "class":self.item_list[idx][1]}



