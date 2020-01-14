import sys
import torch
from utilities.util import uniques, inv_class_frequency, calc_class_frequency
from torch.nn.modules.loss import _WeightedLoss
import numpy as np

def bce(input_tensor, target_tensor, weights=None, class_frequency=False, reduction='sum'):
    # Calculate Class Frequency
    if class_frequency:
        weights = calc_class_frequency(target_tensor)

    input_tensor = torch.nn.functional.sigmoid(input_tensor)

    # print(f"Loss Input {input_tensor.shape}")
    # print(f"Loss Target {target_tensor.shape}")
    # If weights are given or class frequency is activated calculate with weights
    # Add 0.00001 to take into account that a normed matrix will contain 0 and 1
    loss_add = 0.0001
    if weights is not None:
        loss = (target_tensor * torch.log(input_tensor + loss_add)) * weights[0] + \
            ((1 - target_tensor) * torch.log(1 - input_tensor + loss_add)) * weights[1]
    else:
        loss = target_tensor * torch.log(input_tensor + loss_add) \
                + (1 - target_tensor) * torch.log(1 - input_tensor + loss_add)

    loss_r = getattr(torch, reduction)(loss) * -1
    # if loss_r < 0:
    #     print("Type loss {} input {} target {} weights[0] {}".format(loss.type(), input_tensor.type(), target_tensor.type(), weights[0].dtype))
    #     lc = loss.clone().detach().cpu()
    #     ip = input_tensor.clone().detach().cpu()
    #     tt = target_tensor.clone().detach().cpu()
    #     ww = torch.tensor(weights)
    #     import datetime
    #     dt = datetime.datetime.now()
    #     path = "./segmentation/output/info/errors/" + str(dt)
    #     torch.save(lc,path + "loss.pt")
    #     torch.save(ip,path + "input.pt")
    #     torch.save(tt,path + "target.pt")
    #     torch.save(ww, path + "weights.pt")
    #     print("Saved loss of {} (input = [{},{}]; target = [{},{}]) to {}".format(loss_r, torch.min(ip), torch.max(ip), torch.min(tt), torch.max(tt), path))

    assert(not torch.isnan(loss_r))
    assert(not torch.isinf(loss_r))
    return loss_r

class WeightedBinaryCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='sum', class_frequency=False):
        super(WeightedBinaryCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.class_frequency = class_frequency

    def forward(self, input, target):
        return bce(input, target, weights=self.weight, class_frequency=self.class_frequency, reduction=self.reduction)
