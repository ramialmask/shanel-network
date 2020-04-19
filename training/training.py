import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from random import shuffle
from utilities.loaders import load_network, get_loader, get_discriminator_loader
from utilities.util import calc_metrices, split_list, write_meta_dict, get_model_name
import shutil

#TODO Save train, test, validation patches in settings dict!
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
    model_name = get_model_name(settings)    

    # Set up the test/train/val splits
    input_list = os.listdir(settings["paths"]["input_path"])
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
        print(f"Test List {len(test_list[1])} | {sorted(test_list[1][:10])}")
        for train_val_iteration, train_val_list in enumerate(train_val_lists):
            # Create the loaders for training and validation for a network and...
            print(f"Train List {len(train_val_list[0])} | {sorted(train_val_list[0][:10])}")
            print(f"Val List {len(train_val_list[1])} | {sorted(train_val_list[1][:10])}")
            #TODO Dirty, needs fix
            # if settings["network"]  == "classification2d":
            import datetime
            start = datetime.datetime.now()
            print(f"Getting loaders...")
            train_loader = get_discriminator_loader(settings, train_val_list[0])
            train_time = datetime.datetime.now() - start
            start = datetime.datetime.now()
            val_loader = get_discriminator_loader(settings, train_val_list[1])
            val_time = datetime.datetime.now() - start
            print(f"Train Loader {train_time} Val Loader {val_time}")
            # else:
                # train_loader = get_loader(settings, train_val_list[0])
                # val_loader = get_loader(settings, train_val_list[1])

            # ...train the network
            net, val_metrics, val_loss = train(settings, test_iteration, train_val_iteration, epochs, train_loader, val_loader, model_name)
            
            # The network, metrics and loss are saved, allowing for other testing criterias other than loss (eg dice)
            val_candidates.append((net, val_metrics, val_loss))

        # Get the best candidate for a test iteration by lambda sort
        sorted_val_candidates = sorted(val_candidates, key=lambda tu: tu[2])
        best_candidate = sorted_val_candidates[0]
        print(f"Loss of best candidate: {best_candidate[2]}")

        # Test on the best candidate and save the settings
        test_loader = get_discriminator_loader(settings, test_list[1])
        test_score = test(settings, test_iteration, test_loader, best_candidate[0])
        print(f"Test scores {test_score}")
        test_scores.append(test_score)

    # Print the test scores
    result_str = "Test fold;Accuracy;Precision;Recall;Dice;\n"
    for i in range(len(test_scores)):
        result_str += f"{i};{test_scores[i][-2]:.4F};{test_scores[i][0]:.4F};{test_scores[i][1]:.4F};{test_scores[i][-1]:.4F};\n"
    print(result_str)

    with open(settings["paths"]["output_model_path"] + "/" +  model_name + f"/test_scores.txt", "x") as file:
        file.write(result_str)

def train(settings, test_fold, val_fold,  epochs, train_loader, val_loader, model_name):
    """Trains and validates one epoch, writes the output to both screen and attached writer and saves the epoch 
    (eg to recover training progress in case of a crash)
    """
    # Load all components beside the data loaders and create a dedicated writer for this model
    net, criterion, optimizer, scheduler = load_network(settings)

    #TODO make this dir explicit in train.json
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
        _write_progress(writer, test_fold, val_fold, epoch, epochs,train_loss, eval_loss, metrics)

        # Save the epoch and optionally delete the last network
        last_model_path = save_epoch(settings, net, epoch, model_name, test_fold, val_fold, last_model_path)
    return net, metrics, eval_loss

def train_epoch(settings, loader, net, optimizer, criterion):
    """Trains one epoch
    """
    net.train()
    torch.cuda.synchronize()

    # Train loss is saved in order to supervise training progress
    loss_list = []
    import datetime

    for item in loader:
        # Load Volume
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        item_input  = item["volume"]
        input_load_time = datetime.datetime.now() - start
        print(f"\t Input load \t\t{input_load_time}")

        # Load Volume to GPU
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        item_input  = item_input.cuda()
        input_cuda_time = datetime.datetime.now() - start
        print(f"\t Input cuda \t\t{input_cuda_time}")

        # Load Label (to GPU) 
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        item_label  = torch.FloatTensor(item["class"]).cuda()
        label_cuda_time = datetime.datetime.now() - start
        print(f"\t Label cuda \t\t{label_cuda_time}")
        
        # Zero gradients
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        optimizer.zero_grad()
        optimizer_zero_time = datetime.datetime.now() - start
        print(f"\t Optimizer Zero Grad \t{optimizer_zero_time}")
        
        # Forward pass
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        probabilities = net(item_input)
        net_time = datetime.datetime.now() - start
        print(f"\t Net forward \t\t{net_time}")
        
        # Probability shaping
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        probabilities = probabilities.view(-1)
        prob_time = datetime.datetime.now() - start
        print(f"\t Prob view \t\t{prob_time}")
        
        # Loss calculation
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        training_loss = criterion(probabilities, item_label)
        train_loss_time  = datetime.datetime.now() -start
        print(f"\t Train Loss Calc \t{train_loss_time}")

        # Loss backward
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        training_loss.backward()
        train_loss_back_time = datetime.datetime.now() - start
        print(f"\t Train Loss Back \t{train_loss_back_time}")
        
        # Optimizer step
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        optimizer.step()
        optimizer_step_time = datetime.datetime.now() - start
        print(f"\t Optimizer Step \t{optimizer_step_time}")
        
        # Loss detach
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        training_loss = training_loss.detach()#training_loss.data # #float(f"{training_loss}".replace("tensor(","").replace(")",""))
        detach_time = datetime.datetime.now() - start
        print(f"\t Detach \t\t{detach_time}")
        
        # Loss to CPU
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        training_loss = training_loss.cpu()
        cpu_time = datetime.datetime.now() - start
        print(f"\t CPU \t\t\t{cpu_time}")
        
        # Loss to Numpy
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        training_loss = training_loss.numpy()
        np_time = datetime.datetime.now() - start
        print(f"\t NP \t\t\t{np_time}")

        # Append loss to Loss-list
        start = datetime.datetime.now()
        torch.cuda.synchronize()
        loss_list.append(training_loss)
        append_time  = datetime.datetime.now() - start
        print(f"\t Append \t\t{append_time}")
        # print(f"Network time:\n \
        #         \t Input load \t\t{input_load_time}\n \
        #         \t Input cuda \t\t{input_cuda_time}\n \
        #         \t Label cuda \t\t{label_cuda_time}\n \
        #         \t Optimizer Zero Grad \t{optimizer_zero_time}\n \
        #         \t Net forward \t\t{net_time}\n \
        #         \t Prob view \t\t{prob_time}\n \
        #         \t Train Loss Calc \t{train_loss_time}\n \
        #         \t Train Loss Back \t{train_loss_back_time}\n \
        #         \t Optimizer Step \t{optimizer_step_time}\n \
        #         \t Detach \t\t{detach_time}\n \
        #         \t CPU \t\t\t{cpu_time}\n \
        #         \t NP \t\t\t{np_time}\n \
        #         \t Append \t\t{append_time}", end="\r",flush=True)
        print("_"*40 + "\n")
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

    return metric_list, np.average(loss_list)# [np.average(m) for m in metric_list]
    
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

    # metric_list = calc_metrices([r[0] for r in result_list], [r[1] for r in result_list])
    a = [r[0] for r in result_list]
    b = [r[1] for r in result_list]
    metric_list = calc_metrices(a, b)
    #TODO Quick and dirty fix
    return metric_list# [np.average(m) for m in metric_list]

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
