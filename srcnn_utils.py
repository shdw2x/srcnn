import os
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import globals

#region Checkpoint-related Methods
def save_checkpoint(net, optimizer, loss, epoch):
    filename = 'checkpoint_epoch_{}.pt'.format(epoch)
    path = globals.ARGS.outputfolder
    fullpath = os.path.join(path, filename)
    checkpoint = {
                  'network': net, # network = model = structure, includes network parameters
                  'optimizer': optimizer,
                  'loss': loss,
                  'epoch': epoch
                 }
    torch.save(checkpoint, fullpath)
    print("* Network state was saved on checkpoint: {}.".format(fullpath))

def load_checkpoint(path):
    checkpoint = torch.load(path)
    net, optimizer, loss, epoch = checkpoint.values()
    override_config(net, optimizer)
    return (net, optimizer, loss, epoch)
#endregion

#region Output Folder Construction Methods
def construct_output_folder():
    output_root = globals.OUTPUT_ROOT # outputs/
    output_folder_name = globals.ARGS.outputfolder # outputs/output_folder_name

    # Check if outputs directory exists, if not create the directory
    if not os.path.exists(output_root):
        os.mkdir(output_root)

    # Check if output folder name is provided from console, if not create a default name
    if not output_folder_name:
        output_folder_name = create_output_folder_name()

    # Output folder path
    output_folder_path = output_root + output_folder_name + "/"
    globals.ARGS.outputfolder = output_folder_path

    # Check if output_folder_name exists, if not create one, otherwise abort
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    else:
        print("Directory already exists: {}".format(output_folder_path))
        exit(1)

    return output_folder_path

def create_output_folder_name():
    """Returns the name of the folder"""
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
#endregion

#region Image Manipulation Methods
def bicubic_interpolation(image, scale_factor):
    """Takes PIL image, returns up or downscaled image using bicubic interpolation"""
    width, height = image.size
    new_size = (int(width*scale_factor), int(height*scale_factor))
    scaled_image = image.resize(new_size, Image.BICUBIC)
    return scaled_image

def modulation_crop(img, scale_factor):
    """Crop image by a few pixels to make the resolution a multiple of scale_factor"""
    w, h = img.size
    new_w, new_h = w - (w % scale_factor), h - (h % scale_factor)
    cropped_img = img.crop((0, 0, new_w, new_h))
    return cropped_img

def from_tensor_to_PIL(tensor):
    return transforms.functional.to_pil_image(tensor)

def from_PIL_to_tensor(image):
    return transforms.functional.to_tensor(image) # to_tensor: From PIL to Tensor, normalizes values automatically (from docs)
#endregion

#region Image Loading and Transform
class TrainValidationImageTransformer(torchvision.datasets.ImageFolder):
    """A version of the ImageFolder dataset class, customized for the super-resolution task"""

    def __init__(self, root):
        super(TrainValidationImageTransformer, self).__init__(root, transform=None)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (grayscale_image, color_image) where grayscale_image is the decolorized version of the color_image.
        """
        rgb_image, _ = super(TrainValidationImageTransformer, self).__getitem__(index) # ImageFolder.__getitem__: returns PIL object in RGB mode (from docs)

        # Getting the path of each image
        path = self.imgs[index][0]

        y, cb, cr = rgb_image.convert('YCbCr').split()
        scale_factor = globals.ARGS.scalefactor

        # Before downscaling, remove a few pixel from the border to prevent floating point resolutions (i.e. 253px -> 250px -> 125px -> 250px)
        target = modulation_crop(y, scale_factor)
        y_downscaled = bicubic_interpolation(target, 1/scale_factor)
        input = bicubic_interpolation(y_downscaled, scale_factor) # input image is now in the same dimension as target
        return from_PIL_to_tensor(input), from_PIL_to_tensor(target), path

class TestImageTransformer(torchvision.datasets.ImageFolder):
    """A version of the ImageFolder dataset class, customized for the super-resolution task"""

    def __init__(self, root):
        super(TestImageTransformer, self).__init__(root, transform=None)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (grayscale_image, color_image) where grayscale_image is the decolorized version of the color_image.
        """
        rgb_image, _ = super(TestImageTransformer, self).__getitem__(index) # ImageFolder.__getitem__: returns PIL object in RGB mode (from docs)

        # Getting the path of each image
        path = self.imgs[index][0]

        y, cb, cr = rgb_image.convert('YCbCr').split()
        scale_factor = globals.ARGS.scalefactor

        input = bicubic_interpolation(y, scale_factor)
        return from_PIL_to_tensor(input), path

def get_loaders(device, **kwargs):
    """Function to read dataset"""
    load_train = kwargs.get('load_train', False) 
    load_test = kwargs.get('load_test', False)
    batch_size = globals.ARGS.batchsize
    data_root = globals.DATA_ROOT
    loaders = {}
    
    if load_train:
        train_set = TrainValidationImageTransformer(root=data_root + 'train')
        loaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_set = TrainValidationImageTransformer(root=data_root + 'validation')
        loaders['validation'] = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    if load_test:
        test_set = TestImageTransformer(root=data_root + 'test')
        loaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    return loaders
#endregion

#region Configuration-related Methods
def override_config(net, optimizer):
    # Configurations to be overridden
    new_config = {
        "convlayers": 0,
        "kernelsizes": [],
        "kernelcounts": [],
        "relupositions": [],
        "learningrates": []
    }

    # Get hyperparameter settings from the network
    for i, layer in enumerate(net.layers):
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            new_config["convlayers"] += 1
            new_config["kernelsizes"].append(layer.kernel_size[0])
            new_config["kernelcounts"].append(layer.out_channels)
        elif isinstance(layer, torch.nn.modules.activation.ReLU):
            new_config["relupositions"].append(i)

    # Get learning rates from optimizer
    for param_group in optimizer.param_groups:
        new_config["learningrates"].append(param_group['lr'])

    # Replace console values with loaded checkpoint
    config = vars(globals.ARGS)
    for key, item in new_config.items():
        config[key] = item

def get_current_config():
    """Return a string indicating current parameter configuration"""
    config = vars(globals.ARGS)
    message = "\nParameter settings:\n"
    separator = "-" * (len(message)-2) + "\n"
    lines = ""
    for key, item in config.items():
        lines += "- {}: {}\n".format(key, item)
    return (message + separator + lines + separator)

def show_current_config():
    """Print current parameter configuration"""
    print(get_current_config())
    checkpoint_path = globals.ARGS.checkpoint
    if checkpoint_path:
        print("* Network state was loaded from checkpoint: {}".format(checkpoint_path))
        print("* WARNING: Some hyperparameters might be overridden.\n")

def write_current_config(path):
    with open(path + "network_config.txt", 'w+') as file:
        file.write(get_current_config())
    print("Configuration saved.")
#endregion

#region Visualization Methods
def tensorshow(tensor, cmap=None):
    img = from_tensor_to_PIL(tensor)
    if cmap is not None:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)

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

def save_visualized_image_trio(mode, epoch, iteri, loss, psnr, input, pred, target, path):
    draw_image_frequency = globals.DRAW_IMAGE_FREQUENCY

    # Visualize and save image results (input, pred, target) periodically according to frequency value (in epochs)
    if (draw_image_frequency == 1) or (epoch % draw_image_frequency == 0):
        if not globals.SAVED_PICS[mode]:
            globals.SAVED_PICS[mode] = path
        if globals.SAVED_PICS[mode] == path:
            image_name = globals.ARGS.outputfolder + "{}_{}".format(mode, epoch)
            visualize_trio(input, pred, target, path, image_name)
#endregion

def compute_psnr(mse_loss, max_val=1): # max_val is assumed as 1 because of normalization, pass as argument if needed
    return 20.0 * np.log10(max_val/mse_loss)

def console_log(mode, epoch, iteri, loss, psnr):
    print_message_frequency = globals.PRINT_MESSAGE_FREQUENCY
    
    # Print every print_message_frequency mini-batches
    if (not iteri % print_message_frequency):
        print('- %s: [Epoch: %d, Iteration: %3d] Loss: %.4f PSNR: %.4f' % (mode, epoch, iteri, loss, psnr))