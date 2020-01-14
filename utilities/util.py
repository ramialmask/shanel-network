import numpy as np
import torch
from random import shuffle

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
    new_shape = [(ps -of*2)*int(d) for ps, of, d in zip(patches4d.shape[-3:], offset, divs)]
    volume3d = np.zeros(new_shape, dtype=patches4d.dtype)
    shape = volume3d.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    index = 0
    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                patch = patches4d[index]
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
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)

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
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
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

