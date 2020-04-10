import os
import numpy as np
import torch
import json
import nibabel as nib
from random import shuffle
import datetime

def find_divs(settings, volume):
    """Find divs for @get_volume_from_patches3d according to the blocksize
    """
    shape = volume.shape
    mb_size = int(settings["dataloader"]["block_size"])
    return tuple(s / m for s,m in zip(shape,(mb_size, mb_size, mb_size)))

def get_patch_data3d(volume3d, divs=(3,3,6), offset=(6,6,6), seg=False):
    """Generate minibatches, by Giles Tetteh
    Args:
        - volume3d (np.array)       :   The volume to cut
        - divs (tuple, optional)    :   Amount to divide each side
        - offset (tuple, optional)  :   Offset for each div
    """
    if "torch" in str(type(volume3d)):
        volume3d = volume3d.numpy()
    patches = []
    shape = volume3d.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    patch_shape = [w+o*2 for w, o in zip(widths, offset)]
    #print("V3dshape {}".format(volume3d.shape))
    patch_mean = np.mean(volume3d)
    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                patch = np.ones(patch_shape, dtype=volume3d.dtype) * patch_mean
                x_s = x - offset[0] if x - offset[0] >= 0 else 0
                x_e = x + widths[0] + offset[0] if x + \
                        widths[0] + offset[0] <= shape[0] else shape[0]
                y_s = y - offset[1] if y - offset[1] >= 0 else 0
                y_e = y + widths[1] + offset[1] if y + \
                        widths[1] + offset[1] <= shape[1] else shape[1]
                z_s = z - offset[2] if z - offset[2] >= 0 else 0
                z_e = z + widths[2] + offset[2] if z + \
                        widths[2] + offset[2] <= shape[2] else shape[2]

                vp = volume3d[x_s:x_e,y_s:y_e,z_s:z_e]
                px_s = offset[0] - (x - x_s)
                px_e = px_s + (x_e - x_s)
                py_s = offset[1] - (y - y_s)
                py_e = py_s + (y_e - y_s)
                pz_s = offset[2] - (z - z_s)
                pz_e = pz_s + (z_e - z_s)
                patch[px_s:px_e, py_s:py_e, pz_s:pz_e] = vp
                patches.append(patch)

    return torch.tensor(np.array(patches, dtype = volume3d.dtype))

def get_volume_from_patches3d(patches4d, divs = (3,3,6), offset=(0,0,0)):
    """Reconstruct the minibatches, by Giles Tetteh
    Keep offset of (0,0,0) for fully padded volumes
    """
    if "torch" in str(type(patches4d)):
        patches4d = patches4d.numpy()
    # print(f"Patches 4d shape {patches4d.shape}")
    new_shape = [(ps -of*2)*int(d) for ps, of, d in zip(patches4d.shape[-3:], offset, divs)]
    volume3d = np.zeros(new_shape, dtype=patches4d.dtype)
    shape = volume3d.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    index = 0
    # print(f"Shape {shape} widths {widths}")
    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                # print(f"X {x} Y {y} Z {z} index {index} {patches4d.shape}")
                patch = patches4d[index,:,:,:]
                index = index + 1
                volume3d[x:x+widths[0],y:y+widths[1],z:z+widths[2]] = \
                        patch[offset[0]:offset[0] + widths[0], offset[1]:offset[1]+widths[1], offset[2]:offset[2]+widths[2]]
    return torch.tensor(volume3d)

def inv_class_frequency(input_tensor):
    un = uniques(input_tensor)
    if len(un[0]) > 1:
        weights = [float(un[1][0]) / np.prod(input_tensor.size(), dtype=np.float),
            float(un[1][1]) / np.prod(input_tensor.size(), dtype=np.float)]
        return weights
    else:
        return [0,0]

def calc_class_frequency(i_t):
    input_tensor = i_t.clone().detach().cpu()
    n_total = input_tensor.view(-1).size()[0]
    n_FG    = input_tensor.sum().numpy()
    n_BG    = (1-input_tensor).sum().numpy()
    assert(n_total == n_FG + n_BG)  # will fail for non-binary input volumes
    eps = 0.00001
    weights = [n_total/(n_FG + eps), n_total/(n_BG + eps)]

    return weights

def uniques(volume):
    """Extending torchs uniques method to perform like numpy.uniques(count=True)
    Args:
        volume (torch.tensor)   :   Input volume
    Returns:
        list (torch.tensor)     :   Unique values in the input volume
        list (int)              :   Count of unique values
    """
    volume_uniques = volume.unique(sorted=True)
    vl = [(volume == i).sum() for i in volume_uniques]
    volume_uniques_count = torch.stack(vl)
    return volume_uniques.type(volume.type()), volume_uniques_count.type(volume.type())

def calc_statistics(pred, target):
    """Calculate test metrices
    Args:
        - pred      (torch.tensor)  : The predicted values in binary (0,1) format
        - target    (torch.tensor)  : The target value in a binary(0,1) format
    Returns:
        - tp        (int)           : True poitives
        - tn        (int)           : True negatives
        - fp        (int)           : False positives
        - fn        (int)           : False negatives
    """
    # pred = np.asarray(pred).astype(bool)
    # target = np.asarray(target).astype(bool)

    la = lambda a, b: np.logical_and(a, b)
    no = lambda a: np.logical_not(a)

    tp = la(pred, target).sum()
    tn = la(no(pred), no(target)).sum()
    fp = la(pred, no(target)).sum()
    fn = la(no(pred), target).sum()
    return tp, tn, fp, fn

def calc_metrices(pred, target):
    """Calculate test statistics
    Args:
        - pred      (torch.tensor)  : The predicted values in binary (0,1) format
        - target    (torch.tensor)  : The target value in a binary(0,1) format
    Returns:
        - precision (double)        : Precision
        - recall    (double)        : Recall
        - vs        (double)        : Volumetric Similarity
        - accuracy  (double)        : Accuracy score
        - f1_dice   (double)        : Dice/F1-Score
    """
    pred = np.asarray(pred)
    if pred[-1].shape != pred[0].shape:
        pred = pred[:-1]
    pred = pred.astype(bool)
    target = np.asarray(target)
    if target[-1].shape != target[0].shape:
        target = target[:-1]
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        precision = 1
        recall = 1
        vs = 1
        accuracy = 1
        f1_dice = 1
    else:
        tp, tn, fp, fn = calc_statistics(pred, target)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn)
        vs = 1 - abs(fn - fp) / (2*tp + fp + fn)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        f1_dice = (2*tp) / (2*tp + fp +fn)
    return precision, recall, vs, accuracy, f1_dice

def get_model_name(settings):
    model_name = settings["paths"]["output_folder_prefix"] + " " + \
            settings["network"] + " " + settings["training"]["optimizer"]["class"] + \
            " factor " + settings["training"]["scheduler"]["factor"] + " " + \
            settings["training"]["loss"]["class"] + " LR=" + settings["training"]["optimizer"]["learning_rate"] + \
            " Blocksize " + settings["dataloader"]["block_size"] + \
            " Epochs " + settings["training"]["epochs"] + " "+ " | " + str(datetime.datetime.now())

    return model_name

def progress_bar(pars,toto,prefixmsg="", postfixmsg=""):
    percent = 100 * (pars / toto)# + 1
    bars = int(np.floor(percent)) * "█"
    print(prefixmsg + "\t | Overall: %d%% \t" % percent + bars + postfixmsg, end="\r", flush=True)

def split_list(input_list, split_rate):
    split_size = int(np.ceil(len(input_list) * split_rate))
    split_amount = int(len(input_list) / split_size)
    shuffle(input_list)
    result_list = []
    for iteration in range(split_amount):
        small_split = input_list[iteration*split_size:(iteration+1)*split_size]
        big_split = [i for i in input_list if not i in small_split]
        result_list.append((big_split, small_split))
    return result_list

def read_nifti(path):
    """
    volume = read_nifti(path)

    Reads in the NiftiObject saved under path and returns a Numpy volume.
    """
    if(path.find(".nii")==-1):
        path = path + ".nii"
    NiftiObject = nib.load(path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj,0,1)
    return volume

def write_nifti(path,volume):
    """
    write_nifti(path,volume)
    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path. Taken from Olivers filehandling class
    """
    if(path.find(".nii.gz")==-1):
        path = path + ".nii.gz"
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume, 0, 1), affine=affmat)
    nib.save(NiftiObject, os.path.normpath(path))

def read_meta_dict(path, mode):
    """Load the meta dict / settings dict according to the mode
    """
    # Load path dict (always needed)
    settings = {}
    paths_path = os.path.join(path, "paths.json")
    with open(paths_path) as file:
        settings["paths"] = json.loads(file.read())
    
    if mode == "train":
        train_path = os.path.join(path, "train.json")
        with open(train_path) as file:
            settings["training"] = json.loads(file.read())
    elif mode == "predict":
        predict_path = os.path.join(path, "predict.json")
        with open(predict_path) as file:
            settings["prediction"] = json.loads(file.read())

    if mode == "count":
        partition_path = os.path.join(path, "partitioning.json")
        with open(partition_path) as file:
            settings["partitioning"] = json.loads(file.read())
    else:
        network_path = os.path.join(path, "network.json")
        with open(network_path) as file:
            _temp = json.loads(file.read())
            settings["computation"] = _temp["computation"]
            settings["preprocessing"] = _temp["preprocessing"]
            settings["dataloader"] = _temp["dataloader"]
            settings["network"] =  _temp["network"]
            settings["prediction"] = _temp["prediction"]
            settings["postprocessing"] =  _temp["postprocessing"]

    return settings

def write_meta_dict(path, settings, mode="train"):
    path_dir = os.path.join(path, "paths.json")
    with open(path_dir, "w") as file:
        json.dump(settings["paths"], file)

    if mode == "train":
        train_dir = os.path.join(path, "train.json")
        with open(train_dir, "w") as file:
            json.dump(settings["training"], file)
    elif mode == "predict":
        predict_dir = os.path.join(path, "predict.json")
        with open(predict_dir, "w") as file:
            json.dump(settings["prediction"], file)
    if mode == "count":
        partition_path = os.path.join(path, "partitioning.json")
        with open(partition_path, "w") as file:
            json.dump(settings["partitioning"], file)
    else:
        network_path = os.path.join(path, "network.json")
        with open(network_path, "w") as file:
            _temp = {}
            _temp["computation"]    = settings["computation"]
            _temp["preprocessing"]  = settings["preprocessing"]
            _temp["dataloader"]     = settings["dataloader"]
            _temp["network"]        = settings["network"]
            _temp["prediction"]     = settings["prediction"]
            _temp["postprocessing"] = settings["postprocessing"]
            json.dump(_temp, file)
