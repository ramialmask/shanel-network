import torch
import numpy as np

def gen_mask_max_data(data,cutoff=9000):
    """Generate a foreground mask based on high intensities
    """
    data[data > cutoff] = cutoff
    return data

def gen_mask_min_data(data,cutoff=9000):
    """Generate a background mask based on high intensities
    """
    data[data < cutoff] = 0
    return data

def normalize(data):
    """Normalization
    """
    data = data - np.min(data)
    data = data * 1.0 / np.max(data)
    return data

def normalize_std(image):
    eps = 0.0001
    image = image / (np.std(image) + eps) - np.mean(image)
    return image

def histinfo(data, scope="global", cfreq=0.999, min_data=0, max_data=1):
    """Histogram function to locate a significant cut off point
    Args:
        cfreq (float) : Cutoff percentile
    """
    assert(scope=="global" or scope=="local")
    if scope == "local":
        min_data, max_data = np.amin(data), np.amax(data)
    bins = np.arange(np.min(min_data), np.max(max_data), 10)
    vals, bins = np.histogram(data, bins, density=True)
    acc = 0
    cutoff=np.max(bins)
    cfreq *= sum(vals)
    for i, v in enumerate(vals):
        acc = acc+v
        if acc >= cfreq:
            cutoff = bins[i]
            break
    data[data > cutoff] = cutoff
    return data

def _get_neighbourhood(data, x, y, z, neighbourhood):
    lower_bb = [0, 0, 0]
    upper_bb = [0, 0, 0]

    for i, dim in enumerate([x, y, z]):
        if dim - neighourhood < 0:
            lower_bb[i] = 0
        else:
            lower_bb[i] = dim - neighourhood

        if dim + neighourhood >= data.shape[i]:
            upper_bb[i] = data.shape[i] - 1
        else:
            upper_bb[i] = dim + neighourhood
    result = data[lower_bb[0]:upper_bb[0], lower_bb[1]:upper_bb[1], lower_bb[2]:upper_bb[2]]
    return result

def _bernsen_threshold(data, neighbourhood):
    result = np.zeros_like(data)
    for x in range(1, data.shape[0]):
        for y in range(1, data.shape[1]):
            for z in range(1, data.shape[2]):
                #Every single pixel
                neighbourhood = _get_neighbourhood(data, x, y, z, neighbourhood)
                result[x, y, z] = (np.amax(neighbourhood) - np.amin(neighbourhood)) / 2
    return result

def _otsu_threshold(data, neighbourhood):
    return np.zeros_like(data)

def auto_local_threshold(data, neighbourhood=10, method="bernsen"):
    if method == "bernsen":
        return _bernsen_threshold(data, neighbourhood)
    elif method == "otsu":
        return _otsu_threshold(data, neighbourhood)
