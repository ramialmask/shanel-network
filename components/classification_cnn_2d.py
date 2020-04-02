import torch
import torch.nn as nn

""" ClassificationCNN
    (C) Phillip Kaeß """

class ClassificationCNN_2D(nn.Module):
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
        super(ClassificationCNN_2D, self).__init__()
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
        
class ClassificationCNN_2D_FCN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters1=2, num_filters2=4, num_filters3 = 8, num_filters4 = 16, num_filters_rest = 32, kernel_size=7,
                 stride_conv=1, weight_scale=1, pool=2, stride_pool=2, hidden_dim=100, #weight_scale = 0.001
                 num_classes=10, dropout=0.0):
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
        super(ClassificationCNN_2D_FCN, self).__init__()
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
        padding = int((kernel_size - 1) / 2)

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=num_filters1, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)
        self.conv1.weight.data.mul_(weight_scale)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=pool, stride=stride_pool)

        height1 = int((height - kernel_size + 2 * padding + 1) / pool) #=150
        width1 = int((height - kernel_size + 2 * padding + 1) / pool)
        
        self.conv2 = nn.Conv2d(in_channels=num_filters1, out_channels=num_filters2, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)
        self.conv2.weight.data.mul_(weight_scale)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=pool, stride=stride_pool)

        height2 = int((height1 - kernel_size + 2 * padding + 1) / pool)#=75
        width2 = int((height1 - kernel_size + 2 * padding + 1) / pool)
        
        self.conv3 = nn.Conv2d(in_channels=num_filters2, out_channels=num_filters3, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)
        self.conv3.weight.data.mul_(weight_scale)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=pool, stride=stride_pool)

        height3 = int((height2 - kernel_size + 2 * padding + 1) / pool)#=37
        width3 = int((height2 - kernel_size + 2 * padding + 1) / pool)
        
        self.conv4 = nn.Conv2d(in_channels=num_filters3, out_channels=num_filters4, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)
        self.conv4.weight.data.mul_(weight_scale)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=pool, stride=stride_pool)

        height4 = int((height3 - kernel_size + 2 * padding + 1) / pool)#=18
        width4 = int((height3 - kernel_size + 2 * padding + 1) / pool)
        
        self.conv5 = nn.Conv2d(in_channels=num_filters4, out_channels=num_filters_rest, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)
        self.conv5.weight.data.mul_(weight_scale)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=pool, stride=stride_pool)

        height5 = int((height4 - kernel_size + 2 * padding + 1) / pool)#=9
        width5 = int((height4 - kernel_size + 2 * padding + 1) / pool)
        
        self.conv6 = nn.Conv2d(in_channels=num_filters_rest, out_channels=num_filters_rest, kernel_size=kernel_size,
                               stride=stride_conv, padding=padding)
        self.conv6.weight.data.mul_(weight_scale)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=pool, stride=stride_pool)

        height6 = int((height5 - kernel_size + 2 * padding + 1) / pool)#=4
        width6 = int((height5 - kernel_size + 2 * padding + 1) / pool)
        

        self.in_features = num_filters_rest * height6 * width6
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=hidden_dim)
        self.dropout1 = nn.Dropout2d(p=dropout)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

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

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)


        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        x = x.view(-1, num_features)
        # x = x.view(-1,self.in_features)

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        logits = self.fc2(x)
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

