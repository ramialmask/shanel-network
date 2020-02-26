import os
import torch
import numpy as np
from utilities.loaders import load_network, get_prediction_loader

import datetime

def prediction(settings):
    torch.cuda.init()
    torch.cuda.set_device(0)
    print("Predicting...")
    input_list = sorted(os.listdir(settings["paths"]["input_seg_path"]))

    output_path = settings["paths"]["output_seg_path"] + settings["paths"]["output_folder_prefix"] + " | " + str(datetime.datetime.now())
    os.mkdir(output_path)
    print(f"Created {output_path}")
    settings["paths"]["output_seg_path"] = output_path

    loader, dataset = get_prediction_loader(settings, input_list)

    lenloader = len(loader)
    net = load_network(settings, prediction=True)

    threshold = float(settings["prediction"]["threshold"])
    

    sigmoid = settings["prediction"]["sigmoid"]
    binarize = settings["prediction"]["binarize"]

    print(f"Len Loader {lenloader}")
    for idx, item in enumerate(loader):
        # print(f"Predicting item {idx} of {lenloader}")
        # print(f"Predicting item {idx} of {lenloader}\t", end="\r", flush=True)
        item_input  = item.cuda()

        logits = net(item_input)
        
        if sigmoid:
            propabilities = torch.sigmoid(logits).detach().cpu()
        else:
            propabilities = logits.detach().cpu()
         
        predictions = propabilities.numpy()

        if binarize:
            predictions[predictions >= threshold] = 1
            predictions[predictions < threshold] = 0
    
        dataset.save_item(idx, predictions)
        logits = 0
        propabilities = 0
    print("\nDone.")

