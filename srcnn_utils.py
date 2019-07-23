# These are utility functions / classes that you probably dont need to alter.

# import functools
import os
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from skimage import io, color
from skimage.transform import rescale
from skimage.measure import compare_psnr

def read_image(filename):
    pass

def upscale(image):
    pass

norm_vfunc = np.vectorize(lambda x: (2*x)/255 - 1)
denorm_vfunc = np.vectorize(lambda x: (255*x + 255) / 2)

def tensorshow(tensor,cmap=None):
    img = transforms.functional.to_pil_image(tensor/2+0.5)
    if cmap is not None:
        plt.imshow(img,cmap=cmap)
    else:
        plt.imshow(img)

#TODO: Change
class ImageFolder(torchvision.datasets.ImageFolder):
    """A version of the ImageFolder dataset class, customized for the super-resolution task"""

    def __init__(self, root, device):
        super(ImageFolder, self).__init__(root, transform=None)
        self.device = device

    def prepimg(self,img):
        return (transforms.functional.to_tensor(img)-0.5)*2 # normalize tensorized image from [0,1] to [-1,+1]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (grayscale_image, color_image) where grayscale_image is the decolorized version of the color_image.
        """
        color_image,_ = super(ImageFolder, self).__getitem__(index) # Image object (PIL)
        grayscale_image = torchvision.transforms.functional.to_grayscale(color_image)
        return self.prepimg(grayscale_image).to(self.device), self.prepimg(color_image).to(self.device)

def visualize_batch(inputs,preds,targets,save_path=''):
    inputs = inputs.cpu()
    preds = preds.cpu()
    targets = targets.cpu()
    plt.clf()
    bs = inputs.shape[0] if inputs.shape[0] < 5 else 5
    for j in range(bs):
        plt.subplot(3,bs,j+1)
        assert(inputs[j].shape[0]==1)
        tensorshow(inputs[j],cmap='gray')
        plt.subplot(3,bs,bs+j+1)
        tensorshow(preds[j])
        plt.subplot(3,bs,2*bs+j+1)
        tensorshow(targets[j])
    if save_path is not '':
        plt.savefig(save_path)
    else:
        plt.show(block=True)

def save_stats(filename, stats, **kwargs):
    pass

def load_stats(path):
    pass

def plot_train_val_loss(train_losses, val_losses, **kwargs):
    # Epoch lengths
    l_train = len(train_losses)
    l_val = len(val_losses)
    if (not l_train) or (not l_val):
        print("WARNING: No data to plot loss vs epoch!")
        return

    combined = kwargs.get("combined", True)
    save = kwargs.get("save", True)
    show = kwargs.get("show", True)

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Loss vs Epoch Graph')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")

    try:
        # Plot for training
        train_epochs = range(1, l_train+1)
        prepare_plot(ax, train_losses, train_epochs, 'blue', 'train')
        # If separate figures desired
        if not combined:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        # Plot for validation
        val_freq = int(l_train / l_val)
        val_epochs = range(val_freq, l_train+1, val_freq)
        prepare_plot(ax, val_losses, val_epochs, 'orange', 'validation')
    except Exception as e:
        print("WARNING: " + str(e))
        return

    if save:
        path = kwargs.get("path", None)
        if path:
            fig.savefig(path + "/" + 'l_vs_e_plot.png')
        else:
            fig.savefig('l_vs_e_plot.png')
            print("WARNING: Invalid path to save, plot saved under current directory!")
    if show:
        plt.show()

def plot_psnr(accuracies, train_epoch, **kwargs):
    l_acc = len(accuracies)
    if (not l_acc):
        print("WARNING: No data to plot accuracy vs epoch!")
        return

    save = kwargs.get("save", True)
    show = kwargs.get("show", True)

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Accuracy vs Epoch Graph')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")

    eval_freq = int(train_epoch / l_acc)
    eval_epochs = range(eval_freq, train_epoch+1, eval_freq)

    try:
        prepare_plot(ax, accuracies, eval_epochs, 'red', 'accuracy')
    except Exception as e:
        print("WARNING: " + str(e))
        return

    if save:
        path = kwargs.get("path", None)
        if path:
            fig.savefig(path + "/" + 'a_vs_e_plot.png')
        else:
            fig.savefig('a_vs_e_plot.png')
            print("WARNING: Invalid path to save, plot saved under current directory!")
    if show:
        plt.show()

def prepare_plot(ax, losses, epochs, color, label):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(epochs, losses, "o-", color='tab:'+color, label=label)
    ax.grid(True)
    ax.legend()

def compute_psnr(mse_loss, max_val=255):
    return 20.0 * np.log10(max_val/mse_loss)