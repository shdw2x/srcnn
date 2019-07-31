# import functools
import os
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import globals


"""
#TODO: 
Loss plot
PSNR plot
PSNR loss print
Visualize batch side by side
Print the name of the images (input, pred, target)
PSNR for current epoch: 62.0725 (wrong value)
"""

def bicubic_interpolation(image, scale_factor):
    """Takes PIL image, returns up or downscaled image using bicubic interpolation"""
    width, height = image.size
    new_size = (int(width*scale_factor), int(height*scale_factor))
    scaled_image = image.resize(new_size, Image.BICUBIC)
    return scaled_image

def compute_psnr(mse_loss, max_val=255):
    return 20.0 * np.log10(max_val/mse_loss)

norm_vfunc = np.vectorize(lambda x: (2*x)/255 - 1)
denorm_vfunc = np.vectorize(lambda x: (255*x + 255) / 2)

def modulation_crop(img, scale_factor):
    """Crop image by a few pixels to make the resolution a multiple of scale_factor"""
    w, h = img.size
    new_w, new_h = w - (w % scale_factor), h - (h % scale_factor)
    cropped_img = img.crop((0, 0, new_w, new_h))
    return cropped_img

def tensorshow(tensor, cmap=None):
    img = transforms.functional.to_pil_image(tensor)
    if cmap is not None:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)

#TODO: Cb, Cr need to be added later
class ImageFolder(torchvision.datasets.ImageFolder):
    """A version of the ImageFolder dataset class, customized for the super-resolution task"""

    def __init__(self, root):
        super(ImageFolder, self).__init__(root, transform=None)

    def to_tensor(self, img):
        return transforms.functional.to_tensor(img) # to_tensor: Normalizes automatically (from docs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (grayscale_image, color_image) where grayscale_image is the decolorized version of the color_image.
        """
        rgb_image, _ = super(ImageFolder, self).__getitem__(index) # ImageFolder.__getitem__: returns PIL object in RGB mode (from docs)

        # Getting the path of each image
        path = self.imgs[index][0]

        y, cb, cr = rgb_image.convert('YCbCr').split()
        target = modulation_crop(y, globals.ARGS.scalefactor)
        y_downscaled = bicubic_interpolation(target, 1/globals.ARGS.scalefactor)
        input = bicubic_interpolation(y_downscaled, globals.ARGS.scalefactor) # y_upscaled
        return self.to_tensor(input), self.to_tensor(target), path

# Function to read dataset
def get_loaders(device, **kwargs):
    load_train = kwargs.get('load_train', False) 
    load_test = kwargs.get('load_test', False)
    batch_size = globals.ARGS.batchsize
    data_root = globals.DATA_ROOT
    loaders = {}
    
    if load_train:
        train_set = ImageFolder(root=data_root + 'train')
        loaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_set = ImageFolder(root=data_root + 'validation')
        loaders['validation'] = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    if load_test:
        test_set = ImageFolder(root=data_root + 'test')
        loaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return loaders

def visualize_trio(input, pred, target, path, save_path=''):
    trio = [input.cpu(), pred.cpu(), target.cpu()]

    plt.rcParams["figure.figsize"] = [12.0, 8.0]
    plt.clf()

    titles = globals.TITLE
    plt.suptitle(path)

    # Draws input-pred-target plot
    for i in range(len(trio)):
        plt.subplot(1, 3, i+1)
        tensorshow(trio[i])
        plt.title(titles[i])
        plt.axis('off')

    # Subplot Adjust
    plt.subplots_adjust(wspace=0.35)

    if save_path is not '':
        plt.savefig(save_path)
    else:
        plt.show(block=False)

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

def get_current_config():
    """Return a string indicating current parameter configuration"""
    config = vars(globals.ARGS)
    message = "\nParameter settings:\n"
    separator = "-" * (len(message)-2) + "\n"
    lines = ""
    for item, key in config.items():
        lines += "- {}: {}\n".format(item, key)
    return (message + separator + lines + separator)

def show_current_config():
    """Print current parameter configuration"""
    print(get_current_config())

def write_current_config(path):
    with open(path + "config.txt", 'w+') as file:
        file.write(get_current_config())
    print("Configuration saved.")

# Returns the name of the folder
def create_output_folder_name():
    output_root = globals.OUTPUT_ROOT
    folder_names = os.listdir(output_root)

    # Search for folder names including "exp"
    exp_folder_names = list(filter(lambda folder_name: "exp" in folder_name, folder_names))

    if not exp_folder_names:
        return "exp_1"
        
    # Sort folder names with respect to numbers appended
    exp_folder_names.sort(key=lambda f: int(str().join(filter(str.isdigit, f))))

    # Get the name of the folder with highest number
    recent_folder_name = exp_folder_names[-1]

    # Partition the name into two ("exp", "Digits")
    parts = recent_folder_name.split('_')

    # Create folder name exp_digits
    next_folder_name = parts[0] + '_' + str(int(parts[1]) + 1)
    return next_folder_name