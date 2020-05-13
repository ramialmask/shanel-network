import torch
import numpy as np

class EncodingLayer(torch.nn.Module):
    def __init__(self, features_in, features_out, kernel_size=5, stride=1):
        super(EncodingLayer, self).__init__()
        padding = 0 #(kernel_size - 1) // 2
        self.encode = torch.nn.Sequential(
            torch.nn.Conv2d(features_in, features_out, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(features_out, eps=1e-4),
            torch.nn.LeakyReLU()
        )
    def forward(self, x):
        y = self.encode(x)
        return y

class Simple_CNN_2D(torch.nn.Module):
    """
    Leaner code, same network
    """
    def __init__(self, input_dim=(1, 400, 400), kernel_size=3, num_classes=1):
        super(Simple_CNN_2D, self).__init__()
        channels, height, width = input_dim
        self.conv1_4   = EncodingLayer( 1, 4)
        self.conv4_16   = EncodingLayer( 4, 16)
        self.conv16_32   = EncodingLayer( 16, 32)
        num_conv_layers = 3
        img_size = 2
        max_features = 32
        in_features = ((input_dim[1] - num_conv_layers * img_size * 2)**2) * max_features
        self.in_features = in_features
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.conv1_4(x)
        x = self.conv4_16(x)
        x = self.conv16_32(x)

        # Form a vector from matrix
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        x = x.view(-1, num_features)
        
        logits = self.fc1(x)
        probabilities = torch.sigmoid(logits)
        return probabilities
    def save_model(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        # print('Saving model... %s' % path)
        torch.save(self, path)
