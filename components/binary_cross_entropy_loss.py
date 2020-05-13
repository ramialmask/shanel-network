import sys
import torch
from utilities.util import uniques
from torch.nn.modules.loss import _WeightedLoss
import numpy as np

def bce(input_tensor, target_tensor, reduction='sum'):
    # Calculate Class Frequency
    # input_tensor = torch.sigmoid(input_tensor)

    # If weights are given or class frequency is activated calculate with weights
    # Add 0.00001 to take into account that a normed matrix will contain 0 and 1
    loss_add = 0.0001
    loss = target_tensor * torch.log(input_tensor + loss_add) \
            + (1 - target_tensor) * torch.log(1 - input_tensor + loss_add)

    loss_r = getattr(torch, reduction)(loss) * -1
    assert(not torch.isnan(loss_r))
    assert(not torch.isinf(loss_r))
    return loss_r

class BinaryCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='sum', class_frequency=False):
        super(BinaryCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.class_frequency = class_frequency

    def forward(self, input, target):
        return bce(input, target, reduction=self.reduction)
