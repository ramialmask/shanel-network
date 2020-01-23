import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from random import shuffle
from utilities.loaders import load_network, get_loader
from utilities.util import calc_metrices, split_list
import datetime

# TODO 
# Save meta dicts
# Implement retraining

def testfold_training(settings):
    torch.cuda.init()
    torch.cuda.set_device(0)
    print("Network:\t" + settings['network'])
    print(f"Loss:\t\t" + settings["training"]["loss"]["class"])
    print(f"Loss reduction\t" + settings["training"]["loss"]["reduction"])
    print(f"Learning rate:\t" + settings["training"]["optimizer"]["learning_rate"])
    print(f"Momentum:\t" + settings["training"]["optimizer"]["momentum"])
    print(f"Optimizer:\t" + settings["training"]["optimizer"]["class"])

    epochs = int(settings["training"]["epochs"])
    model_name =  settings['paths']['output_folder_prefix'] + \
            settings['network'] + ' ' + settings['training']['optimizer']['class'] + \
            ' factor ' + settings['training']['scheduler']['factor'] + ' ' + \
            settings['training']['loss']['class'] + ' LR=' + settings['training']['optimizer']['learning_rate'] + \
            ' Blocksize ' + settings['dataloader']['block_size'] + \
            ' Epochs ' + settings['training']['epochs'] + ' '+ ' | ' + str(datetime.datetime.now())

    input_list = os.listdir(settings['paths']['input_raw_path'])
    test_split_rate = float(settings["training"]["crossvalidation"]["test_split_rate"])
    train_val_split_rate = float(settings["training"]["crossvalidation"]["train_val_split_rate"])
    test_lists = split_list(input_list, test_split_rate)

    test_scores = []

    for test_iteration, test_list in enumerate(test_lists):
        train_val_lists = split_list(test_list[0], train_val_split_rate)
        val_candidates = []
        for train_val_iteration, train_val_list in enumerate(train_val_lists):
            train_loader = get_loader(settings, train_val_list[0])
            val_loader = get_loader(settings, train_val_list[1])

            net, val_metrics, val_loss = train(settings, test_iteration, train_val_iteration, epochs, train_loader, val_loader, model_name)
            val_candidates.append((net, val_metrics, val_loss))
        sorted_val_candidates = sorted(val_candidates, key=lambda tu: tu[2])
        best_candidate = sorted_val_candidates[0]
        print(f"Loss of best candidate: {best_candidate[2]}")
        test_loader = get_loader(settings, test_list[1])
        test_score = test(settings, test_iteration, test_loader, best_candidate[0])
        print(f"Test scores {test_score}")
        test_scores.append(test_score)

    for i in range(len(test_scores)):
        print(f"Test dice score {i}:\t{test_scores[i][-1]}")

def train(settings, test_fold, val_fold,  epochs, train_loader, val_loader, model_name):
    net, criterion, optimizer, scheduler = load_network(settings)
    writer = SummaryWriter(f"/home/ramial-maskari/runs/{model_name}/{test_fold}/{val_fold}")

    last_model_path = ""

    for epoch in range(epochs):
        train_loss = train_epoch(settings, train_loader, net, optimizer, criterion)
        metrics, eval_loss = validate_epoch(settings, val_loader, net, optimizer, criterion)
        print(f"{test_fold} {val_fold} Epoch {epoch} of {epochs}\tTrain Loss:\t{train_loss}\tValidation Loss:\t{eval_loss}\tValidation Dice:\t{metrics[-1]}")

        writer.add_scalar(f'Loss/Training', train_loss, epoch)
        writer.add_scalar(f'Loss/Validation', eval_loss, epoch)
        writer.add_scalar(f'Validation Metrics/Precision', metrics[0], epoch)
        writer.add_scalar(f'Validation Metrics/Recall', metrics[1], epoch)
        writer.add_scalar(f'Validation Metrics/Accuracy', metrics[-2], epoch)
        writer.add_scalar(f'Validation Metrics/Dice', metrics[-1], epoch)
        
        last_model_path = save_epoch(settings, net, epoch, model_name, test_fold, val_fold, last_model_path)
    #TODO write Meta Dict
    return net, metrics, eval_loss

def train_epoch(settings, loader, net, optimizer, criterion):
    net.train()
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
    net.eval()
    threshold = float(settings["prediction"]["threshold"])
    metric_list = []
    loss_list = []
    for item in loader:
        item_input  = item["volume"].cuda()
        item_label  = item["segmentation"].cuda()

        logits = net(item_input)
        val_loss = criterion(logits, item_label)
        propabilities = torch.sigmoid(logits).detach().cpu().numpy()
        
        predictions = propabilities
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        metric_list.append(calc_metrices(predictions, item_label.detach().cpu().numpy()))
        loss_list.append(val_loss.detach().cpu().numpy())

    
    return [np.average(m) for m in metric_list], np.average(loss_list)
    
def test(settings, test_iteration, loader, net):
    net.eval()
    metric_list = []
    threshold = float(settings["prediction"]["threshold"])
    for item in loader:
        item_input  = item["volume"].cuda()
        item_label  = item["segmentation"].cuda()

        logits = net(item_input)
        propabilities = torch.sigmoid(logits).detach().cpu().numpy()
        
        predictions = propabilities
        predictions[predictions >= threshold] = 1
        predictions[predictions < threshold] = 0

        metric_list.append(calc_metrices(predictions, item_label.detach().cpu().numpy()))

    return [np.average(m) for m in metric_list]

def save_epoch(settings, net, epoch, model_name, test_fold, val_fold, last_model_path):
    if settings["training"]["delete_qs"] == "True" and last_model_path != "":
        os.unlink(last_model_path)

    model_save_dir = os.path.join(settings["paths"]["output_model_path"], model_name)
    model_save_path = os.path.join(model_save_dir, settings["paths"]["model_name"],f"_{test_fold}_{val_fold}_{epoch}.dat")
    
    print(f"Saving model to {model_save_path} in {model_save_dir}...")
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    net.save_model(model_save_path)
    print("Saved model.")
    return model_save_path
