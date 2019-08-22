import torch
import torch.nn as nn
from srcnn_utils import get_pad_count

import globals

# SRCNN
class SRNET(nn.Module):
    def __init__(self):
        # Base class initialization
        super(SRNET, self).__init__() #TODO: Search super (with parameters)

        self.layers = []
        calculate_padding = lambda f: int(get_pad_count(f))

        # Adding layers
        # Adding first convolutional layer
        kernel_size = globals.ARGS.kernelsizes[0]
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=globals.ARGS.kernelcounts[0], 
                                     kernel_size=kernel_size, padding=calculate_padding(kernel_size), bias=True))

        # Adding remaining convolutional layers
        for i in range(1, globals.ARGS.convlayers):
            kernel_size = globals.ARGS.kernelsizes[i]
            self.layers.append(nn.Conv2d(in_channels=globals.ARGS.kernelcounts[i-1], out_channels=globals.ARGS.kernelcounts[i], 
                                         kernel_size=kernel_size, padding=calculate_padding(kernel_size), bias=True))
              
        # Adding ReLUs
        for i in globals.ARGS.relupositions:
            self.layers.insert(i, nn.ReLU())

        # Place layers in a sequence
        self.layers = nn.Sequential(*self.layers)
        print(self.layers)

    def forward(self, image):   
        pred = self.layers(image)
        return pred