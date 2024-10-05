# UNET
Implementation of a U-Net using PyTorch for semantic segmentation ([Dataset](https://www.kaggle.com/datasets/mantasu/face-synthetics-glasses)). This repository consist in 5 python files:

  1. **glasses.py** : Converts the local directory with the images to a PyTorch dataset.
  2. **unet.py** : Implementation of the U-Net in PyTorch.
  3. **processing.py** : Functions to process the data, and some utils.
  4. **train.py** : Train the U-Net and obtain training and validation losses.
  5. **test.py** : Test the U-Net on the test set.

The *U-Net* is a Convolutional Network that was proposed for biomedical segmentation in 2015. The aim for this repository is to replicate the arquitecture and study its application for semantic segmentation.

The arquitecture of the U-Net consist in a encoder part, with double 2D convolutions and 2x2 MaxPools, a bottleneck, with a single double 2D convolution, and the decoder part, with 2D transposed convolutions and double convolutions.
At the end a 1x1 convolution is applied to map all the channels into an output with $$channels = classes$$.

In the *losses* folder some plots of the training and validation costs vs epochs can be found. The *model* folder contains the weights of the model. The *results* folder consist of pictures of the test set along with the original masks, and the predicted masks by the model.



***Notes***:
  * The accuracy metrics haven't been implemented yet.
  * As this isn't the final version of the repository, there is an appreciable absence of comments in code.

