import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torchvision as tv
from torchsummary import summary
import time
from processing import *
from unet import *

img_size = 64
batch_size = 64
epochs = 5
n_channels = 3
n_classes = 1
lr = 0.007

train_set, val_set, test_set = sets(image_size = img_size)
m_train, m_val, m_test = len(train_set), len(val_set), len(test_set)

train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle = True)

model_path = "model/UNET_weights.pth"

model = UNET(n_channels, n_classes)
model.load_state_dict(torch.load(model_path, weights_only = True))
print("Model weights loaded!")

samples = 5

fig, ax = plt.subplots(samples, 3)

for idx, (image, mask) in enumerate(test_loader):
    pred = torch.sigmoid(model(image))
    pred = return_PIL(pred)
    image = return_PIL(image)
    mask = return_PIL(mask)

    ax[idx, 0].xaxis.set_tick_params(labelbottom=False) 
    ax[idx, 0].yaxis.set_tick_params(labelleft=False)
    ax[idx, 1].xaxis.set_tick_params(labelbottom=False) 
    ax[idx, 1].yaxis.set_tick_params(labelleft=False) 
    ax[idx, 2].xaxis.set_tick_params(labelbottom=False) 
    ax[idx, 2].yaxis.set_tick_params(labelleft=False)   

    if idx == 0:
        ax[idx, 0].set_title("Image")
        ax[idx, 1].set_title("Mask")
        ax[idx, 2].set_title("Predicted Mask")

    ax[idx, 0].imshow(image)
    ax[idx, 1].imshow(mask, cmap = "gray")
    ax[idx, 2].imshow(pred, cmap = "gray")

    if idx == samples-1:
        plt.savefig(f"results/results4.pdf")
        plt.show()
        break


