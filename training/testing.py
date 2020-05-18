import os
import torch
import numpy as np
import sys
sys.path.append("/home/ramial-maskari/Documents/Pytorch Network")
from utilities.loaders import load_network, get_loader, get_discriminator_loader
from utilities.util import calc_metrices,read_meta_dict
from create_figure import create_summary
import shutil
import pandas as pd

def testfold_training(settings, model_path, df):
    """Splits the training data into test folds, train folds and validation folds
    according to the meta information in train.json and trains a multitude of networks.
    The progress is written to a tensorboard. Models with the best validation score of 
    a validation fold will be tested.
    """
    # Initialize the gpu
    gpu = settings["computation"]["gpu"]
    torch.cuda.init()
    torch.cuda.set_device(gpu)


    # Print core training stats
    print("Network:\t" + settings["network"])
    print(f"Loss:\t\t" + settings["training"]["loss"]["class"])
    print(f"Loss reduction\t" + settings["training"]["loss"]["reduction"])
    print(f"Learning rate:\t" + settings["training"]["optimizer"]["learning_rate"])
    print(f"Momentum:\t" + settings["training"]["optimizer"]["momentum"])
    print(f"Optimizer:\t" + settings["training"]["optimizer"]["class"])

    test_list = settings["training"]["crossvalidation"]["test_set"]

    net = torch.load(model_path)


    # Test on the best candidate and save the settings
    test_loader = get_discriminator_loader(settings, test_list)
    test_score, mcl, dataframe  = test(settings, test_loader, net, df)
    print(test_score)
    return mcl, dataframe

def test(settings, loader, net, df):
    """Tests an epoch and calculates precision, recall, accuracy, volumetric similarity and f1-score
    """
    net.eval()
    # Load the binarization threshold
    threshold = float(settings["prediction"]["threshold"])
    # Test metrics are saved in order to supervise training progress
    result_list = []

    # List of missclassified patches
    missclassified_list = []
    
    for item_no, item in enumerate(loader):
        item_input  = item["volume"].cuda()
        item_label  = item["class"].cuda()

        propabilities = net(item_input)
        propabilities = propabilities.view(-1)
        propabilities = propabilities.detach().cpu().numpy()
        propabilities_float = float(propabilities[0])
        

        item_item = item["item"]
        item_axis = item["axis"].numpy()
        
        # if propabilities > 0.1 and propabilities < 0.9:
        #     print(f"{item_item} {item_axis} {propabilities}")
        
        # Stick to proper naming...
        predictions = propabilities
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        # Detach everything
        item_label = item_label.detach().cpu().numpy()
        item_input = item_input.detach().cpu().numpy()

        # Fill the dataframe
        df_item = pd.DataFrame({"patch":[item_item[0]],\
                                "axis":[int(item_axis[0])],\
                                "class":[float(item_label[0])],\
                                "predicted class":[float(predictions[0])],\
                                "false classified":predictions != item_label,\
                                "propability":[propabilities_float]})
    
        df = df.append(df_item)


        if predictions != item_label:
            missclassified_list.append((item_item,item_axis))
            # print(f"{item_no}\t\t{predictions}\t\t{item_label}\t\t{propabilities}\t\t{item_axis}\t\t{item_item}")

        result_list.append([predictions, item["class"].numpy()])

    a = [r[0] for r in result_list]
    b = [r[1] for r in result_list]
    metric_list = calc_metrices(a, b)
    return metric_list, missclassified_list, df


output_folder = "/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/input/false classified 20 05 05 subsample/"
model_name = "2D Classification test sample 20-05-05 subset safe leanclassification2d Adam factor 0.1 WBCELoss LR=1e-4 Blocksize 100 Epochs 50  | 2020-05-11 19:10:19.058385"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
import shutil

best_models = [(0, 3),(1,3),(2,2),(3,2)]

# Create empty dataframe
df = pd.DataFrame(columns=['patch','axis','class','predicted class','false classified','propability'])
for model in best_models:
    test_fold = model[0]
    var_fold = model[1]
    settings_path = f"/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/output/models/{model_name}/{test_fold}/{var_fold}/"
    settings = read_meta_dict(settings_path, "train")
    input_folder = settings["paths"]["input_path"]
    model_path = settings_path + f"_{test_fold}_{var_fold}_49.dat"
    mcl, df = testfold_training(settings, model_path, df)
df.to_csv(f"/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/output/models/{model_name}/test.csv")

path = f"/media/ramial-maskari/16TBDrive/Synthetic Neuron Creation/classification/output/models/{model_name}/"
image_input_path = settings["paths"]["input_path"]

create_summary(df, path, image_input_path)
    
    # for misclassified_patch in mcl:
    #     patch_name = misclassified_patch[0][0]
    #     patch_axis = misclassified_patch[1][0]
    #     print(f"Moving {patch_name}_{patch_axis}.tiff")
    #     input_item_path = os.path.join(input_folder, patch_name, f"{patch_name}_{patch_axis}.tiff")
    #     output_item_path = os.path.join(output_folder, f"{patch_name}_{patch_axis}.tiff")
    #     shutil.copy(input_item_path, output_item_path)
