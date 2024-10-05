import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torchvision as tv


class GlassesDataset(torch.utils.data.Dataset):
    """
    Glasses dataset for semantic segmentation.
    """
    
    def __init__(self, root, transformation = None):
        """
        Arguments:
        root -- root of the dataset (train, val, test)
        transformation -- torch tranformations applyed to the dataset
        """

        self.root = root
        self.transformation = transformation
        self.images_root = root + "images/"
        self.masks_root = root + "masks/"
        self.images_list = os.listdir(self.images_root)
        self.masks_list = os.listdir(self.masks_root)
    

    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, idx):
        """
        Given an index, returns a dictionary with the corresponding image and mask.

        Arguments: 
        idx -- int, index from the dataset

        Returns:
        sample -- dictionary, image and mask for corresponding index
        """

        image = Image.open(self.images_root + self.images_list[idx])
        mask = Image.open(self.masks_root + self.masks_list[idx]).convert("L")


        if self.transformation:
            image = self.transformation(image)
            mask = self.transformation(mask)

        return image, mask
    
    def shape(self):
        image = Image.open(self.images_root + self.images_list[0])
        tensor = self.transformation(image)
        return tensor.shape  