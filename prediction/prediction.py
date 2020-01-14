import os
import torch
import numpy as np
from utilities.loaders import load_network, get_prediction_loader


def prediction(settings):
    print("Predicting...")
    input_list = os.listdir(settings['paths']['input_raw_path'])

    print(f"Input list len {len(input_list)}")
    loader, dataset = get_prediction_loader(settings, input_list)

    print(f"len loader {len(loader)}")
    net, _, _, _ = load_network(settings)

    threshold = float(settings["prediction"]["threshold"])

    for idx, item in enumerate(loader):
        print(len(item))
        item_input  = item.cuda()

        logits = net(item_input)
        propabilities = torch.sigmoid(logits).detach().cpu().numpy()
        
        predictions = propabilities
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0
    
        dataset.save_item(idx, predictions)
        # TODO save predictiona

