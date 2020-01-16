import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from functools import partial
from components.deep_vessel_3d import Deep_Vessel_Net_FC
from components.unet_3d_oliver import Unet3D
from components.weighted_binary_cross_entropy_loss import WeightedBinaryCrossEntropyLoss
from components.dice_loss import DiceLoss
from components.training_dataset import TrainingDataset
from components.prediction_dataset import PredictionDataset

def get_optimizer(settings, net):
    """Get a optimizer object
    """
    lr = settings["training"]["optimizer"]["learning_rate"]
    momentum = settings["training"]["optimizer"]["momentum"]
    optimizer_class = settings["training"]["optimizer"]["class"]

    # print(f"Learning rate:\t{lr}")
    # print(f"Momentum:\t{momentum}")
    # print(f"Optimizer:\t{optimizer_class}")

    if optimizer_class == "SGD":
        if lr == "Default" and momentum == "Default":
            optimizer = optim.SGD(net.parameters())
        elif lr == "Default" and momentum != "Default":
            optimizer = optim.SGD(net.parameters(),
                        momentum=float(momentum))
        elif lr != "Default" and momentum == "Default":
            optimizer = optim.SGD(net.parameters(),
                        lr=float(lr))
        else:
            optimizer = optim.SGD(net.parameters(),
                        lr=float(lr),
                        momentum=float(momentum))
    elif optimizer_class == "Adam":
        if lr == "Default":
            optimizer = optim.Adam(net.parameters())
        else:
            optimizer = optim.Adam(net.parameters(),
                    lr=float(lr))
    elif optimizer_class == "Amsgrad":
        if lr == "Default":
            optimizer = optim.Adam(net.parameters(), amsgrad=True)
        else:
            optimizer = optim.Adam(net.parameters(),
                        lr=float(lr),
                        amsgrad=True)

    return optimizer

def get_lr_optim(settings, optimizer):
    """ Get a learning rate scheduler
    """
    lr = optimizer.state_dict()["param_groups"][0]["lr"]
    factor = float(settings["training"]["scheduler"]["factor"])
    patience = int(settings["training"]["scheduler"]["patience"])
    min_factor = float(settings["training"]["scheduler"]["min_factor"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",factor=factor, patience=patience, threshold_mode="abs", min_lr=lr*min_factor, verbose=True)
    return scheduler

def get_loss(settings):
    """Get a loss object
    """
    loss = settings["training"]["loss"]["class"]
    loss_reduction = settings["training"]["loss"]["reduction"]
    # print(f"Loss:\t\t{loss}")
    # print(f"Loss reduction\t{loss_reduction}")
    if loss == "MSELoss":
        criterion = nn.MSELoss()
    elif loss == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif loss == "BCELoss":
        criterion = nn.BCELoss()
    elif loss == "WBCELoss":
        cf = False
        if settings["training"]["loss"]["weight"] == "CF":
            cf = True
        print("Using Class Frequency: {}".format(str(cf)))
        criterion = WeightedBinaryCrossEntropyLoss(class_frequency=cf, reduction=loss_reduction)
    elif loss == "WBCELossLogits":
        criterion = nn.BCEWithLogitsLoss(reduction=loss_reduction, pos_weight=torch.tensor([0.99,0.01]))
    elif loss == "DiceLoss":
        criterion = DiceLoss()
    else:
        criterion = "MSELoss"
    return criterion

def load_network(settings):
    """Load a network object and loss function and optimizer 
    Args:
        - settings (dict)   : The global settings dict
    """
    if settings['network'] == 'deepvesselnet':
        net = Deep_Vessel_Net_FC(settings)
    elif settings['network'] == 'unet3d':
        net = Unet3D(settings)

    # TODO
    # if testing or prediction or settings['training']['retraining'] == "True":
    #     model_path = settings['paths']['input_model_path'] + settings['paths']['input_model']
    #     t_ = torch.load(model_path)
    #     net.load_state_dict(t_)
    if settings["computation"]["use_cuda"] == "True":
        net = net.cuda()

    criterion = get_loss(settings)
    optimizer = get_optimizer(settings, net)

    scheduler = get_lr_optim(settings, optimizer)

    return net, criterion, optimizer, scheduler

def load_trained_network(settings):
    """Load a network object
    Args:
        - settings (dict)   : The global settings dict
    """

    if settings['network'] == 'deepvesselnet':
        net = Deep_Vessel_Net_FC(settings)
    elif settings['network'] == 'unet3d':
        net = Unet3D(settings)

    model_path = settings['paths']['input_model_path'] + settings['paths']['input_model']
    t_ = torch.load(model_path)
    net.load_state_dict(t_)
    if settings["computation"]["use_cuda"] == "True":
        net = net.cuda()


def get_loader(settings, input_list):
    item_dataset = TrainingDataset(settings, input_list)
    item_len = len(item_dataset)
    item_batch_size = item_len + 1
    if (item_len + 1) % int(settings["dataloader"]["batch_size"]) == 0 or item_batch_size > 5:
        item_batch_size = int(settings["dataloader"]["batch_size"])
    item_params = {
        "num_workers":int(settings["dataloader"]["num_workers"]),
        "pin_memory":True,
        "batch_sampler":BatchSampler(
            RandomSampler(item_dataset),
            batch_size=item_batch_size,
            drop_last=(settings["dataloader"]["drop_last"] == "True"))
    }
    item_loader = DataLoader(item_dataset, **item_params)
    return item_loader

def get_prediction_loader(settings, input_list):
    item_dataset = PredictionDataset(settings, input_list)
    item_len = len(item_dataset)
    item_batch_size = item_len + 1
    if (item_len + 1) % int(settings["dataloader"]["batch_size"]) == 0 or item_batch_size > 5:
        item_batch_size = int(settings["dataloader"]["batch_size"])
    item_params = {
        "num_workers":int(settings["dataloader"]["num_workers"]),
        "pin_memory":True,
        "batch_sampler":BatchSampler(
            SequentialSampler(item_dataset),
            batch_size=item_batch_size,
            drop_last=False)
    }
    item_loader = DataLoader(item_dataset, **item_params)
    return item_loader, item_dataset
#def load_loader(settings, input_list, iteration=0):
#    val_split, train_split = get_splits(settings,input_list, iteration) 
#    train_dataset   = TrainingDataset(settings,train_split)#, norm=partial(norm_data_whole_body, settings=settings))
#    val_dataset     = TrainingDataset(settings, val_split)#, norm=partial(norm_data_whole_body, settings=settings))

#    #Use +1 to avoid loosing one item due to drop_last
#    train_len = len(train_dataset)
#    train_batch_size = train_len + 1
#    if (train_len + 1) % int(settings["dataloader"]["batch_size"]) == 0 or train_batch_size > 5:
#        train_batch_size = int(settings["dataloader"]["batch_size"])
#    print("Train Dataloader Batchsize: {}".format(train_batch_size))


#    val_len = len(val_dataset)
#    val_batch_size = val_len + 1
#    if (val_len + 1) % int(settings["dataloader"]["batch_size"]) == 0 or val_batch_size > 5:
#        val_batch_size = int(settings["dataloader"]["batch_size"])
#    print("Validation Dataloader Batchsize: {}".format(val_batch_size))

#    train_params = {
#        "num_workers":int(settings["dataloader"]["num_workers"]),
#        "pin_memory":True,
#        "batch_sampler":BatchSampler(
#            RandomSampler(train_dataset),
#            batch_size=train_batch_size,#int(settings["dataloader"]["batch_size"]), # 12
#            drop_last=(settings["dataloader"]["drop_last"] == "True"))
#    }

#    val_params = {
#        "num_workers":int(settings["dataloader"]["num_workers"]),
#        "pin_memory":True,
#        "batch_sampler":BatchSampler(
#            RandomSampler(val_dataset),
#            batch_size=val_batch_size,#int(settings["dataloader"]["batch_size"]), # 5
#            drop_last=(settings["dataloader"]["drop_last"] == "True"))
#    }
#    train_loader = DataLoader(train_dataset, **train_params)
#    val_loader = DataLoader(val_dataset, **val_params)
#    return train_loader, val_loader 

def get_splits(settings, input_list, iteration=0):
    """ Get splits for a given iteration
    If no test split should occur, set training - crossvalidation - test_split_rate to -1
    """
    # Input has to be already shuffled!
    split_rate = float(settings["training"]["crossvalidation"]["train_val_split_rate"])

    lenlist = len(input_list)

    fold_size = int(np.around(lenlist * split_rate))

    smaller_split = input_list[lenlist - iteration *
    fold_size - fold_size:lenlist - iteration*fold_size]

    print(f"\t\tSMALLER SPLIT {smaller_split}")

    bigger_split = [i for i in input_list if i not in smaller_split]

    assert(len(set(smaller_split) & set(bigger_split)) == 0)


    return smaller_split, bigger_split
