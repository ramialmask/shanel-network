import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from random import shuffle
from utilities.loaders import load_network, get_loader, get_discriminator_loader
from utilities.util import calc_metrices, split_list, read_meta_dict, write_meta_dict, get_model_name
from utilities.create_figure import create_summary
import shutil
import pandas as pd

def crossvalidation(settings):
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

    early_stopping = settings["training"]["early_stopping"] == "True"
    if early_stopping:
        assert(settings["training"]["delete_qs"] == "False")

    # Concatenate the model name
    epochs = int(settings["training"]["epochs"])
    model_name = get_model_name(settings)    

    # Set up the test/train/val splits
    input_list = os.listdir(settings["paths"]["input_path"])
    test_split_rate = float(settings["training"]["crossvalidation"]["test_split_rate"])
    train_val_split_rate = float(settings["training"]["crossvalidation"]["train_val_split_rate"])
    test_lists = split_list(input_list, test_split_rate)

    # Final test scores are saved through all folds, initialize outside of loop
    df = pd.DataFrame(columns=["Test Fold","Validation Fold", "Epoch", "Train Loss", "Validation Loss", "Validation Accuracy", "Validation Precision", "Validation Recall", "Validation Dice"])

    for test_iteration, test_list in enumerate(test_lists):
        # For each test itearation, the train/validation splits are fixed
        train_val_lists = split_list(test_list[0], train_val_split_rate)
        # Validation scores as well as the corresponding networks are saved in this list
        print(f"Test List {len(test_list[1])} | {sorted(test_list[1][:10])}")
        for train_val_iteration, train_val_list in enumerate(train_val_lists):
            # Create the loaders for training and validation for a network and...
            print(f"Train List {len(train_val_list[0])} | {sorted(train_val_list[0][:10])}")
            print(f"Val List {len(train_val_list[1])} | {sorted(train_val_list[1][:10])}")
            train_loader = get_discriminator_loader(settings, train_val_list[0])
            val_loader = get_discriminator_loader(settings, train_val_list[1])
            
            settings["training"]["crossvalidation"]["training_set"] = train_val_list[0]
            settings["training"]["crossvalidation"]["validaton_set"] = train_val_list[1]
            settings["training"]["crossvalidation"]["test_set"] = test_list[1]

            # ...train the network
            net, val_metrics, val_loss, df = train(settings, test_iteration, train_val_iteration, epochs, train_loader, val_loader, model_name, df)

    if early_stopping:
        test_df = early_stopping(settings, df, model_name)
    else:
        test_df = test_crossvalidation(settings, df, model_name)

    model_path = settings["paths"]["output_model_path"] + model_name
    df.to_csv(f"{model_path}/training.csv")
    test_df.to_csv(f"{model_path}/test.csv")
    create_summary(test_df, model_path, settings["paths"]["input_path"])

def test_crossvalidation(settings, df, model_name):
    """Calculate the best epoch for each test fold and compute the score of the best model
    Test scores are saved in test.csv
    """
    test_columns=["Test Fold", "Validation Fold", "Test Accuracy", "Test Precision", "Test Recall", "Test Dice"]
    test_df = pd.DataFrame(columns=test_columns)

    test_folds = range(0, int(1 / float(settings["training"]["crossvalidation"]["test_split_rate"])))
    val_folds  = range(0, int(1 / float(settings["training"]["crossvalidation"]["train_val_split_rate"])))

    epoch = int(settings["training"]["epochs"]) - 1

    model_path = settings["paths"]["output_model_path"] + model_name

    min_val_loss = 90001
    best_fold = -1

    for test_fold in test_folds:
        # For each of the models get best validation loss
        for val_fold in val_folds:
            df_fold = df.loc[(df["Test Fold"] == test_fold) & (df["Validation Fold"] == val_fold) & (df["Epoch"] == epoch)]
            if df_fold["Validation Loss"][0] < min_val_loss:
                min_val_loss = df_fold["Validation Loss"][0]
                best_fold = df_fold

        best_val_fold = best_fold["Validation Fold"]
        best_model_path = os.path.join(model_path, str(test_fold), str(val_fold))
        best_model_data_path = best_model_path + f"/_{test_fold}_{val_fold}_{epoch}.dat"
        
        # Once we have the best model path, we need to update the settings to get the correct test folds
        settings = read_meta_dict(best_model_path, "train")
        best_model, _, _, _ = load_network(settings, model_path = best_model_data_path)

        # Test on the best candidate and save the settings
        test_list = settings["training"]["crossvalidation"]["test_set"]
        test_loader = get_discriminator_loader(settings, test_list)
        test_score = test(settings, test_fold, test_loader, best_model)
        print(f"Test scores {test_score}")
        test_item = pd.DataFrame({"Test Fold":[test_fold],\
                            "Validation Fold":[best_val_fold],\
                            "Test Accuracy":[test_score[0]],\
                            "Test Precision":[test_score[1]],\
                            "Test Recall":[test_score[-2]],\
                            "Test Dice":[test_score[-1]],\
                            })
        test_df.append(test_item)
    return test_df

def early_stopping(settings, df, model_name):
    """Calculate the best epoch for each test fold and compute the score of the best model
    Test scores are saved in test.csv
    """
    test_columns=["Test Fold", "Validation Fold", "Epoch", "Test Accuracy", "Test Precision", "Test Recall", "Test Dice"]
    test_df = pd.DataFrame(columns=test_columns)

    test_folds = range(0, int(1 / float(settings["training"]["crossvalidation"]["test_split_rate"])))
    val_folds  = range(0, int(1 / float(settings["training"]["crossvalidation"]["train_val_split_rate"])))


    model_path = settings["paths"]["output_model_path"] + model_name

    for test_fold in test_folds:
        # Find the epoch with the overall lowest val score in one val fold
        best_epochs = []
        for val_fold in val_folds:
            df_fold = df.loc[(df["Test Fold"] == test_fold) & (df["Validation Fold"] == val_fold)]
            min_val = df_fold["Validation Loss"].min()
            best_epoch = df_fold.loc[(df_fold["Validation Loss"] == min_val)]["Epoch"]
            best_epochs.append(best_epoch)
        val_epoch = int(np.mean(best_epochs))
        min_val_loss = 9000
        best_fold = -1
        # For each of the models get best validation loss
        for val_fold in val_folds:
            df_fold = df.loc[(df["Test Fold"] == test_fold) & (df["Validation Fold"] == val_fold) & (df["Epoch"] == val_epoch)]
            if df_fold["Validation Loss"][0] < min_val_loss:
                min_val_loss = df_fold["Validation Loss"][0]
                best_fold = df_fold

        best_val_fold = best_fold["Validation Fold"]
        best_epoch = best_fold["Epoch"][0]
        best_model_path = os.path.join(model_path, str(test_fold), str(val_fold))
        best_model_data_path = best_model_path + f"/_{test_fold}_{val_fold}_{best_epoch}.dat"
        
        # Once we have the best model path, we need to update the settings to get the correct test folds
        settings = read_meta_dict(best_model_path, "train")
        best_model, _, _, _ = load_network(settings, model_path = best_model_data_path)

        # Test on the best candidate and save the settings
        test_list = settings["training"]["crossvalidation"]["test_set"]
        test_loader = get_discriminator_loader(settings, test_list)
        test_score = test(settings, test_fold, test_loader, best_model)
        print(f"Test scores {test_score}")
        test_item = pd.DataFrame({"Test Fold":[test_fold],\
                            "Validation Fold":[best_val_fold],\
                            "Epoch":[best_epoch],\
                            "Test Accuracy":[test_score[0]],\
                            "Test Precision":[test_score[1]],\
                            "Test Recall":[test_score[-2]],\
                            "Test Dice":[test_score[-1]],\
                            })
        test_df.append(test_item)
    return test_df

def train(settings, test_fold, val_fold,  epochs, train_loader, val_loader, model_name, df):
    """Trains and validates one epoch, writes the output to both screen and attached writer and saves the epoch 
    (eg to recover training progress in case of a crash)
    """
    # Load all components beside the data loaders and create a dedicated writer for this model
    net, criterion, optimizer, scheduler = load_network(settings)

    writer_path = settings["paths"]["writer_path"]
    writer = SummaryWriter(f"{writer_path}{model_name}/{test_fold}/{val_fold}")

    # This variable holds the location of the last trained network so all temporary saved networks can be deleted
    last_model_path = ""

    for epoch in range(epochs):
        # Train one epoch and validate it
        train_loss = train_epoch(settings, train_loader, net, optimizer, criterion)
        metrics, eval_loss = validate_epoch(settings, val_loader, net, optimizer, criterion)
        scheduler.step(eval_loss)
        
        # Write down the progress
        df = _write_progress(writer, test_fold, val_fold, epoch, epochs,train_loss, eval_loss, metrics, df)

        # Save the epoch and optionally delete the last network
        last_model_path = save_epoch(settings, net, epoch, model_name, test_fold, val_fold, last_model_path)
    return net, metrics, eval_loss, df

def train_epoch(settings, loader, net, optimizer, criterion):
    """Trains one epoch
    """
    net.train()
    # torch.cuda.synchronize()

    # Train loss is saved in order to supervise training progress
    loss_list = []

    for item in loader:
        # Load Volume
        item_input  = item["volume"]

        # Load Volume to GPU
        item_input  = item_input.cuda()

        # Load Label (to GPU) 
        item_label  = torch.FloatTensor(item["class"]).cuda()
        
        # # Zero gradients
        optimizer.zero_grad()
        
        # # Forward pass
        probabilities = net(item_input)
        
        # # Probability shaping
        probabilities = probabilities.view(-1)
        
        # # Loss calculation
        training_loss = criterion(probabilities, item_label)

        # # Loss backward
        training_loss.backward()
        
        # # Optimizer step
        optimizer.step()
        
        # # Loss detach
        training_loss = training_loss.detach()
        
        # # Loss to CPU
        training_loss = training_loss.cpu()
        
        # # Loss to Numpy
        training_loss = training_loss.numpy()

        # # Append loss to Loss-list
        loss_list.append(training_loss)
    return np.average(loss_list)

def validate_epoch(settings, loader, net, optimizer, criterion):
    """Validates an epoch and and adjusts the training using the optimizer
    """
    net.eval()
    # Load the binarization threshold
    threshold = float(settings["prediction"]["threshold"])
    # Validation loss and metrics are saved in order to supervise training progress
    result_list = []
    loss_list = []
    for item_no, item in enumerate(loader):
        item_input  = item["volume"].cuda()
        item_label  = item["class"].cuda()

        probabilities = net(item_input)
        probabilities = probabilities.view(-1)
        propabilities = probabilities.detach().cpu().numpy()

        # Get validation loss
        val_loss = criterion(probabilities, item_label)
        
        # Stick to proper naming...
        predictions = propabilities
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        result_list.append([predictions, item["class"].numpy()])
        loss_list.append(val_loss.detach().cpu().numpy())

    a = [r[0] for r in result_list]
    b = [r[1] for r in result_list]
    metric_list = calc_metrices(a, b)

    return metric_list, np.average(loss_list)

def test(settings, test_iteration, loader, net):
    """Tests an epoch and calculates precision, recall, accuracy, volumetric similarity and f1-score
    """
    net.eval()
    # Load the binarization threshold
    threshold = float(settings["prediction"]["threshold"])
    # Test metrics are saved in order to supervise training progress
    result_list = []
    for item in loader:
        item_input  = item["volume"].cuda()
        item_label  = item["class"].cuda()

        propabilities = net(item_input)
        propabilities = propabilities.view(-1)
        propabilities = propabilities.detach().cpu().numpy()
        
        # Stick to proper naming...
        predictions = propabilities
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        result_list.append([predictions, item["class"].numpy()])

    a = [r[0] for r in result_list]
    b = [r[1] for r in result_list]
    metric_list = calc_metrices(a, b)
    return metric_list

def save_epoch(settings, net, epoch, model_name, test_fold, val_fold, last_model_path):
    """Saves an epoch into a new path and deletes the model from the previous epoch
    """
    # If quicksaves should be deleted and there is a quicksave already, delete it
    if settings["training"]["delete_qs"] == "True" and last_model_path != "":
        shutil.rmtree(last_model_path)

    # Create the directory tree where the model and the meta information is saved
    model_save_dir = os.path.join(settings["paths"]["output_model_path"], model_name)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model_save_dir = os.path.join(model_save_dir, str(test_fold))
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model_save_dir = os.path.join(model_save_dir, str(val_fold))
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model_save_path = os.path.join(model_save_dir, settings["paths"]["model_name"] + f"_{test_fold}_{val_fold}_{epoch}.dat")

    # Save the model and the meta information
    net.save_model(model_save_path)
    write_meta_dict(model_save_dir, settings, "train")
    return model_save_dir

def _write_progress(writer, test_fold, val_fold, epoch, epochs, train_loss, eval_loss, metrics, df):
    """Writes the progress of the training both on the default output as well as the connected tensorboard writer
    """

    # Print the test progress to std.out
    print(f"{test_fold} {val_fold} Epoch {epoch} of {epochs}\tTrain Loss:\t{train_loss}\tValidation Loss:\t{eval_loss}\tValidation Dice:\t{metrics[-1]}")
    
    # Construct Dataframe for train.csv
    df_item = pd.DataFrame({"Test Fold":[test_fold],\
                            "Validation Fold":[val_fold],\
                            "Epoch":[epoch],\
                            "Train Loss": [train_loss],\
                            "Validation Loss":[eval_loss],\
                            "Validation Accuracy":[metrics[0]],\
                            "Validation Precision":[metrics[1]],\
                            "Validation Recall":[metrics[-2]],\
                            "Validation Dice":[metrics[-1]],\
                            })
    df = df.append(df_item)
    
    # Write the progress to the tensorboard
    writer.add_scalar(f"Loss/Training", train_loss, epoch)
    writer.add_scalar(f"Loss/Validation", eval_loss, epoch)
    writer.add_scalar(f"Validation Metrics/Precision", metrics[0], epoch)
    writer.add_scalar(f"Validation Metrics/Recall", metrics[1], epoch)
    writer.add_scalar(f"Validation Metrics/Accuracy", metrics[-2], epoch)
    writer.add_scalar(f"Validation Metrics/Dice", metrics[-1], epoch)
    return df
