# SHANEL Prediction & Cell Counting

This repository contains the code for [_Zhao et al: Cellular and Molecular Probing of Intact Human
Organs_](https://doi.org/10.1101/643908) .

## Overview

In this repository you will find the ressources to perform all deep learning and image analysis tasks described in _Zhao et al: Cellular and Molecular Probing of Intact Human
Organs_, including training a convolutional neural network, predicting a binary mask on tissue-cleared images and counting cells on binary masks. In order to train a network on your own, predict a binary mask on a dataset or count the number of cells you only need to manipulate the json files. Each json file contains the necessary (hyper)parameters for a specific area:
- paths.json        : all path informations for training, prediction and counting
- partitioning.json : all information about the partitioning of a scan; cutting down from one large scan to small patches
- network.json      : all network, data loading, pre- and postprocessing information
- train.json        : all information for training a network, including optimizer, loss class and hyperparameter optimization 

## Requirements

- Python 3.6
- numpy
- pytorch 
- torchvision 
- cudatoolkit=10.1
- nibabel
- connected-components-3d
- multiprocessing
- json

## Cell Counting

Cell counting works on binary masks. After predicting your data, set `["paths"]["input_count_path"]` in the _paths.json_ file to the folder containing your segmentation masks and enter `python __main__.py count` in your console. The amount of cells will be written on screen.

## Prediction

After training a model, you will have the network.json file containing most neccessary information to predict on new data, e.g. the name of the network architecture, the correct padding, the right normalization values. Only path variables have to be set in the _paths.json_ file: 

1. Set `["paths"]["input_seg_path"]` to the directory containing the raw items
2. Set `["paths"]["input_model_path"]` to the directory containing your model file
3. Set `["paths"]["output_seg_path"]` to the directory where the new output folder containing all predictions will be created
4. Set `["paths"]["output_folder_prefix"]`, `["paths"]["output_prefix"]`, `["paths"]["output_postfix"]` as you wish
5. Enter `python __main__.py predict` in your console

## Training

- Show some tensorboard pics
- How to access tensorboard
- Which gpu to use
