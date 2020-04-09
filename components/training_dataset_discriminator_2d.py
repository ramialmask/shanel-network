import os

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utilities.util import find_divs, get_patch_data3d
from PIL import Image
"""
2D Implementation of a Training Loader
- Load all datasets into one huge list
- Iterate over this list as method of getitem
"""
class TrainingDatasetDiscriminator_2D(Dataset):
    def __init__(self, settings, split, transform=None, norm=None):
        self.settings = settings

        # Get paths
        image_path = settings["paths"]["input_path"]

        # create list
        image_list = []

        # Load data
        for item in split:
            item_image_path = os.path.join(image_path, item)

            image       = np.array(Image.open(item_image_path)).astype(np.float64)#[:100,:100]
            image_class = -1
            if "raw" in item:
                image_class = 0
            elif "bg" in item:
                image_class = 1
            image_class = np.array(image_class).astype(np.float64)

            if transform:
                image       = transform(image)

            if norm:
                image       = norm(image)
            
            # Torchify
            image       = torch.tensor(image).float().unsqueeze(0)
            image_class = torch.tensor(image_class).float()

            image_list.append((image, image_class))

        self.item_list = image_list

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        return {"volume":self.item_list[idx][0], "class":self.item_list[idx][1]}



