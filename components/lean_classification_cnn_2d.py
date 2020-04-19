import torch
import numpy as np

class EncodingLayer(torch.nn.Module):
    def __init__(self, features_in, features_out, kernel_size=3, stride=1):
        super(EncodingLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = torch.nn.Sequential(
            torch.nn.Conv2d(features_in, features_out, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(features_out, eps=1e-4),
            torch.nn.ReLU()
        )
    def forward(self, x):
        y = self.encode(x)
        return y

class LeanClassificationCNN_2D(torch.nn.Module):
    """
    Leaner code, same network
    """
    def __init__(self, input_dim=(1, 400, 400), kernel_size=3, num_classes=1):
        super(LeanClassificationCNN_2D, self).__init__()
        channels, height, width = input_dim
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_2   = EncodingLayer( 1, 2)
        self.conv2_2   = EncodingLayer( 2, 2)
        self.conv2_4   = EncodingLayer( 2, 4)
        self.conv4_4   = EncodingLayer( 4, 4)
        self.conv4_8   = EncodingLayer( 4, 8)
        self.conv8_8   = EncodingLayer( 8, 8)
        self.conv8_16  = EncodingLayer( 8,16)
        self.conv16_16 = EncodingLayer(16,16)
        num_pooling_layers = 6
        img_size = int(np.floor(height/(2**num_pooling_layers)) * np.floor(width/(2**num_pooling_layers)))
        in_features = channels * img_size * 16
        self.fc1 = torch.nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        x = self.pool(x) # Pooling #1
        x = self.conv2_4(x)
        x = self.conv4_4(x)
        x = self.pool(x) # Pooling #2
        x = self.pool(x) # Pooling #3
        x = self.conv4_8(x)
        x = self.conv8_8(x)
        x = self.pool(x) # Pooling #4
        x = self.pool(x) # Pooling #5
        x = self.conv8_16(x)
        #x = self.conv16_16(x) # --> not present in previous version of code
        x = self.pool(x) # Pooling #6

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
        print('Saving model... %s' % path)
        torch.save(self, path)
