import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from random import shuffle
from utilities.loaders import load_network, get_loader
from utilities.util import calc_metrices, split_list, write_meta_dict, get_model_name
import datetime
import shutil

def testfold_training(settings):
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


    # Concatenate the model name
    epochs = int(settings["training"]["epochs"])
    model_name =  get_model_name(settings)    

    # Set up the test/train/val splits
    input_list = os.listdir(settings["paths"]["input_raw_path"])
    test_split_rate = float(settings["training"]["crossvalidation"]["test_split_rate"])
    train_val_split_rate = float(settings["training"]["crossvalidation"]["train_val_split_rate"])
    test_lists = split_list(input_list, test_split_rate)

    # Final test scores are saved through all folds, initialize outside of loop
    test_scores = []

    for test_iteration, test_list in enumerate(test_lists):
        # For each test itearation, the train/validation splits are fixed
        train_val_lists = split_list(test_list[0], train_val_split_rate)
        # Validation scores as well as the corresponding networks are saved in this list
        val_candidates = []
        for train_val_iteration, train_val_list in enumerate(train_val_lists):
            # Create the loaders for training and validation for a network and...
            train_loader = get_loader(settings, train_val_list[0])
            val_loader = get_loader(settings, train_val_list[1])

            #Save training, validation and test sets to the settings
            settings["training"]["crossvalidation"]["training_set"] = train_val_list[0]
            settings["training"]["crossvalidation"]["validaton_set"] = train_val_list[1]
            settings["training"]["crossvalidation"]["test_set"] = test_list[1]

            # ...train the network
            net, val_metrics, val_loss = train(settings, test_iteration, train_val_iteration, epochs, train_loader, val_loader, model_name)
            
            # The network, metrics and loss are saved, allowing for other testing criterias other than loss (eg dice)
            val_candidates.append((net, val_metrics, val_loss, train_val_iteration))

        # Get the best candidate for a test iteration by lambda sorting for the validation loss
        sorted_val_candidates = sorted(val_candidates, key=lambda tu: tu[2])
        best_candidate = sorted_val_candidates[0]
        print(f"Loss of best candidate: {best_candidate[2]} - Candidate #{best_candidate[3]}")

        # Test on the best candidate and save the settings
        test_loader = get_loader(settings, test_list[1])
        test_score = test(settings, test_iteration, test_loader, best_candidate[0])
        print(f"Test scores {test_score}")
        test_scores.append((best_candidate[3], test_score))

    # Print the test scores
    for i in range(len(test_scores)):
        print(f"Test dice score test iteration #{i} train_val iteration {test_scores[i][0]}:\t{test_scores[i][1][-1]}")

def train(settings, test_fold, val_fold,  epochs, train_loader, val_loader, model_name):
    """Trains and validates one epoch, writes the output to both screen and attached writer and saves the epoch 
    (eg to recover training progress in case of a crash)
    """
    # Load all components beside the data loaders and create a dedicated writer for this model
    net, criterion, optimizer, scheduler = load_network(settings)
    writer = SummaryWriter(f"/home/ramial-maskari/runs/{model_name}/{test_fold}/{val_fold}")

    # This variable holds the location of the last trained network so all temporary saved networks can be deleted
    last_model_path = ""

    for epoch in range(epochs):
        # Train one epoch and validate it
        train_loss = train_epoch(settings, train_loader, net, optimizer, criterion)
        metrics, eval_loss = validate_epoch(settings, val_loader, net, optimizer, criterion)
        scheduler.step(eval_loss)
        
        # Write down the progress
        _write_progress(writer, test_fold, val_fold, epoch, epochs,train_loss, eval_loss, metrics)

        # Save the epoch and optionally delete the last network
        last_model_path = save_epoch(settings, net, epoch, model_name, test_fold, val_fold, last_model_path)
    return net, metrics, eval_loss

def train_epoch(settings, loader, net, optimizer, criterion):
    """Trains one epoch
    """
    net.train()

    # Train loss is saved in order to supervise training progress
    loss_list = []
    for item in loader:
        item_input  = item["volume"].cuda()
        item_label  = item["segmentation"].cuda()

        optimizer.zero_grad()
        logits = net(item_input)

        training_loss = criterion(logits, item_label)
        training_loss.backward()

        optimizer.step()
        loss_list.append(training_loss.clone().detach().cpu().numpy())
    return np.average(loss_list)

def validate_epoch(settings, loader, net, optimizer, criterion):
    """Validates an epoch and and adjusts the training using the optimizer
    """
    net.eval()
    # Load the binarization threshold
    threshold = float(settings["prediction"]["threshold"])
    # Validation loss and metrics are saved in order to supervise training progress
    metric_list = []
    loss_list = []
    for item in loader:
        item_input  = item["volume"].cuda()
        item_label  = item["segmentation"].cuda()

        logits = net(item_input)
        val_loss = criterion(logits, item_label)
        propabilities = torch.sigmoid(logits).detach().cpu().numpy() # XXX
        # TODO Softmax vs sigmoid in settings/network, not here
        # propabilities = F.softmax(logits).detach().cpu().numpy()
        
        # Stick to proper naming...
        predictions = propabilities
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        metric_list.append(calc_metrices(predictions, item_label.detach().cpu().numpy()))
        loss_list.append(val_loss.detach().cpu().numpy())

    
    return [np.average(m) for m in metric_list], np.average(loss_list)
    
def test(settings, test_iteration, loader, net):
    """Tests an epoch and calculates precision, recall, accuracy, volumetric similarity and f1-score
    """
    net.eval()
    metric_list = []
    # Load the binarization threshold
    threshold = float(settings["prediction"]["threshold"])
    for item in loader:
        item_input  = item["volume"].cuda()
        item_label  = item["segmentation"].cuda()

        logits = net(item_input)
        propabilities = torch.sigmoid(logits).detach().cpu().numpy()
        
        # Stick to proper naming...
        predictions = propabilities
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        metric_list.append(calc_metrices(predictions, item_label.detach().cpu().numpy()))

    return [np.average(m) for m in metric_list]

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

def _write_progress(writer, test_fold, val_fold, epoch, epochs, train_loss, eval_loss, metrics):
    """Writes the progress of the training both on the default output as well as the connected tensorboard writer
    """
    print(f"{test_fold} {val_fold} Epoch {epoch} of {epochs}\tTrain Loss:\t{train_loss}\tValidation Loss:\t{eval_loss}\tValidation Dice:\t{metrics[-1]}")
    writer.add_scalar(f"Loss/Training", train_loss, epoch)
    writer.add_scalar(f"Loss/Validation", eval_loss, epoch)
    writer.add_scalar(f"Validation Metrics/Precision", metrics[0], epoch)
    writer.add_scalar(f"Validation Metrics/Recall", metrics[1], epoch)
    writer.add_scalar(f"Validation Metrics/Accuracy", metrics[-2], epoch)
    writer.add_scalar(f"Validation Metrics/Dice", metrics[-1], epoch)
