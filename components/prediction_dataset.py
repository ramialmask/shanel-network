import os

import torch
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset, DataLoader
from utilities.util import find_divs, get_patch_data3d, get_volume_from_patches3d, write_nifti
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
# TODO Switch cases if shape/batchsize-mismatch in save_item
class PredictionDataset(Dataset):
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings
        self.split = split
        # Get paths
        self.nii_path = settings["paths"]["input_raw_path"]

        # create list
        nii_list = []

        mb_size = int(self.settings["dataloader"]["block_size"])

        self.vdivs = -1
        self.offset_volume = -1
        # Load data
        print(f"SPLIT {split}")
        # exit()
        for item in split:
            item_nii_path = os.path.join(self.nii_path, item)
            image = np.swapaxes(nib.load(item_nii_path).dataobj, 0, 1).astype(np.int64)

            if transform:
                image = transform(image)

            if norm:
                image = norm(image)

            # Torchify
            image = torch.tensor(image).float()

            if image.shape[0] > mb_size:
                self.vdivs = find_divs(self.settings, image)
                offset_value = int(self.settings["preprocessing"]["padding"])
                self.offset_volume = (offset_value, offset_value, offset_value)

                image_list = [x for x in get_patch_data3d(image, divs=self.vdivs, offset=self.offset_volume).unsqueeze(1)]

                nii_list = nii_list + image_list
            else:
                nii_list.append(image.unsqueeze(1))

            print(f"{item} {len(nii_list)}")


        # Initiate single list
        self.prediction_buffer_counter = -1
        block_size = int(settings["dataloader"]["block_size"])
        batch_size = int(settings["dataloader"]["batch_size"])
        initial_shape = image.shape
        print(f"Block size {block_size}\nBatch size {batch_size}\nInitial shape {initial_shape}")
        required_batchsize = np.prod([i / block_size for i in initial_shape])
        print(f"BATCHSIZE\t{batch_size}\nREQUIRED\t{required_batchsize}")
        if batch_size < required_batchsize:
            self.required_batchsize = required_batchsize
            self.prediction_buffer = -1
            self.prediction_buffer_size = int(required_batchsize / batch_size)
            self.prediction_buffer_counter = 0
            print(f"Buffer size {self.prediction_buffer_size}")

        self.item_list = nii_list

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        return self.item_list[idx]

    def save_item(self, idx, item):
        batch_size = int(self.settings["dataloader"]["batch_size"])
        output_path = self.settings["paths"]["output_seg_path"]

        if self.prediction_buffer_counter > -1:
            if self.prediction_buffer_counter == 0:
                self.prediction_buffer = item
            else:
                self.prediction_buffer = np.concatenate([self.prediction_buffer, item], axis=0)
            self.prediction_buffer_counter += 1
            if self.prediction_buffer_counter == self.prediction_buffer_size:
                #concat buffer, output image
                self.prediction_buffer = np.concatenate([self.prediction_buffer, item], axis=0)
                item = np.squeeze(self.prediction_buffer)
                direct_item = get_volume_from_patches3d(item, divs=self.vdivs)
                direct_item = direct_item.numpy().astype(np.int32)
                # Required: ratio of required_batchsize 
                item_index = int(idx / self.prediction_buffer_size)
                direct_item_name = self.split[item_index]
                print(f"Saving {idx} | {item_index} | {direct_item_name}") 
                direct_output_path = os.path.join(output_path, direct_item_name)
                write_nifti(direct_output_path, direct_item)
                self.prediction_buffer = -1
                self.prediction_buffer_counter = 0
        else:
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
            write_nifti(direct_output_path, direct_item)



