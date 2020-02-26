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
