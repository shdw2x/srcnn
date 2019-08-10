import os
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.util.shape import view_as_windows

filename_extension_regex = r'(.+)\.(\w+)'
file_index_regex = r'img(\d+)_\d+_\d+\.\w+'

# Renames the image names
def image_rename():
    path = "./dataset/train/"
    images = os.listdir(path)
    start = 1
    rename_lst = []

    for image in images:
        match = re.search(file_index_regex, image)
        if match:
            val = int(match.group(1))
            if val >= start:
                start = val + 1
        else:
            rename_lst.append(image)

    for index, image in enumerate(rename_lst, start):
        im = Image.open(path + image)
        width, height = im.size
        im.close()
        extension = re.search(filename_extension_regex, image).group(2)
        os.rename(path + image, path + "img{i}_{w}_{h}.{e}".format(i=index, w=width, h=height, e=extension))

# Extract patches from an image in the form of Numpy array
def extract_patches(image, patch_shape=(33, 33, 3), stride=14):
    patches = view_as_windows(image, patch_shape, stride)
    row_count, col_count, t, height, width, channel = patches.shape
    patch_count = row_count * col_count
    patches =  np.reshape(patches, (patch_count, height, width, channel))
    return patches

# Save extracted patches one by one
def save_patches(path, name, patches):
    filename = re.search(filename_extension_regex, name).group(1)
    extension = re.search(filename_extension_regex, name).group(2)
    for i, patch in enumerate(patches, 1):
        new_name = filename + "_patch" + str(i)
        plt.imsave(os.path.join(path, new_name), patch)

# Creates and save patches
def create_and_save_patches():
    path = "./dataset/train/images/"
    image_names = os.listdir(path)
    for image_name in image_names[1:]:
        image = np.array(Image.open(path + image_name))
        patches = extract_patches(image)
        save_patches("./patches", image_name, patches)

create_and_save_patches()