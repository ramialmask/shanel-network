import torch
import torch.nn as nn

""" ClassificationCNN in 3D based on the work of Phillip Kaeß"""

class ClassificationCNN_3D(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 300, 300), kernel_size=3,
                 stride_conv=1, pool=2, stride_pool=2,
                 num_classes=1):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN_3D, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # conv kernel
        self.bn2 = torch.nn.BatchNorm2d(2, eps=1e-4)
        self.bn4 = torch.nn.BatchNorm2d(4, eps=1e-4)
        self.bn8 = torch.nn.BatchNorm2d(8, eps=1e-4)

        
        padding = int((kernel_size - 1) / 2)

        self.conv1_2 = nn.Conv2d(in_channels=channels, out_channels=2, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)
        self.relu = nn.ReLU()

        height1 = height#300
        
        self.conv2_2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)

        self.pool = nn.MaxPool2d(kernel_size=pool, stride=stride_pool)

        height2 = int((height1 - kernel_size + 2 * padding + 1) / pool)#=150
        
        
        self.conv2_4 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)


        height3 = int((height2 - kernel_size + 2 * padding + 1) / pool)#=75
        
        self.conv4_4 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)
        
        self.conv4_8 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)

        self.conv8_8 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)

        height4 = int((height3 - kernel_size + 2 * padding + 1) / pool)#=37

        
        self.conv8_16 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)
      

        height5 = int((height4 - kernel_size + 2 * padding + 1) / pool)#=18
        height6 = int((height5 - kernel_size + 2 * padding + 1) / pool)#=9
        
        self.conv16_16 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)
        
        height7 = int((height6 - kernel_size + 2 * padding + 1) / pool)#=4
        height8 = int((height7 - kernel_size + 2 * padding + 1) / pool)#=2

        self.in_features = height7 * height7 # * 16 
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain oiew function to make the     #
        # transition from the spatial input image tur previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.vo the flat fully connected  #
        # layers.                                                              #
        ########################################################################

        x = self.conv1_2(x)#in: 1  out:2  (Params: 3x3x1x2)
        x = self.relu(x)
        
        x = self.conv2_2(x)#in: 2  out:2  (Params: 3x3x2x2)
        x = self.relu(x)
        x = self.bn2(x)
        
        
        x = self.pool(x)#image_size is now 150
    
        x = self.conv2_4(x)#in: 2  out:4  (Params: 3x3x2x4)
        x = self.relu(x)
        
        x = self.conv4_4(x)#in: 4  out:4  (Params: 3x3x4x4)
        x = self.relu(x)
        x = self.bn4(x)

        
        x = self.pool(x)#image_size is now 75
        x = self.pool(x)#image_size is now 37
        
        x = self.conv4_8(x)#in: 4  out:8  (Params: 3x3x4x8)
        x = self.relu(x)
        x = self.bn8(x)
        
        x = self.conv8_8(x)#in: 8  out:8  (Params: 3x3x8x8)
        x = self.relu(x)
        
        
        x = self.pool(x)#image_size is now 18
        x = self.pool(x)#image_size is now 9
        
        x = self.conv8_16(x)#in: 8  out:16 (Params: 3x3x8x16)
        x = self.relu(x)
        x = self.pool(x)#image_size is now 4
        
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        x = x.view(-1, num_features)
        # x = x.view(-1,self.in_features)

        logits = self.fc1(x)
        probabilities = torch.sigmoid(logits)
        return probabilities

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def save_model(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
        
