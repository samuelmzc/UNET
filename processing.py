from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torchvision as tv
from glasses import GlassesDataset


def sets(image_size, root = "archive/glass/"):
    transformations = tv.transforms.Compose([
        tv.transforms.Resize((image_size, image_size)),
        tv.transforms.ToTensor()
    ])

    train_set =  GlassesDataset(root + "train/", transformation = transformations)
    validation_set = GlassesDataset(root + "val/", transformation = transformations)
    test_set = GlassesDataset(root + "test/", transformation = transformations)

    return train_set, validation_set, test_set


def return_PIL(tensor):
    """
    Apply the inversed transformations to obtain a PIL image

    Arguments:
    tensor -- torch.Tensor

    Returns:
    image -- PIL image, tensor converted into an image
    """

    inversed = tv.transforms.Compose([
        tv.transforms.Lambda(lambda img : img.permute(1, 2, 0)),
        tv.transforms.Lambda(lambda img : img*255),
        tv.transforms.Lambda(lambda img : img.detach().numpy().astype(np.uint8)),
        tv.transforms.ToPILImage()
    ])

    if len(tensor.shape) == 4:        # if there are batches, take first batch
        tensor = tensor[0, :, :, :]
    
    image = inversed(tensor)
    return image


def show_samples(dataset, n_samples):
    """
    Plots images and masks from the dataset.

    Arguments:
    dataset -- torch.utils.data.Dataset
    n_samples -- int, nº of samples to show
    """

    fig, axs = plt.subplots(n_samples, 2)

    for idx, (image, mask) in enumerate(dataset):
        image = return_PIL(image)
        mask = return_PIL(mask)    
        
        if idx == 0:
            axs[idx, 0].set_title("Image")
            axs[idx, 1].set_title("Mask")
        else:
            pass
        
        axs[idx, 0].imshow(image)
        axs[idx, 0].set_ylabel(f"Sample {idx + 1}")
        axs[idx, 1].imshow(mask)

        if idx == n_samples - 1:
            plt.savefig("transformed.png")
            plt.show()
            break
    

def plot_losses(train_losses, val_losses, plot = "y"):
    epochs = len(train_losses)
    epoch_arr = np.linspace(1, epochs, epochs)
    plt.plot(epoch_arr, np.array(train_losses), label = "Train loss")
    plt.plot(epoch_arr, np.array(val_losses), label = "Validation loss")
    plt.title("Binary Cross Entropy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{epochs}:eps.pdf")
    if plot.lower() == "y":
        plt.show()


def check_acc(model, loader):
    model.eval()
    hits = 0
    pixels = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            scores = torch.nn.functional.sigmoid(preds)
            scores = (scores > 0.5).float() # boolean mask
            y = (y > 0.5).float()
            hits += (preds == y).sum()
            pixels += torch.numel(scores) # numel = number of elements of tensor

    accuracy = hits/pixels
    return accuracy


def seconds_to_hhmmss(time):
    hours = int(time/3600)
    minutes = int(time/60 - hours*60)
    seconds = int(time - hours*3600 - minutes*60)
    return [hours, minutes, seconds]