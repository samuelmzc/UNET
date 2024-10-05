import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import torchvision as tv
from torchsummary import summary
import time
from processing import *
from unet import *

# Set device where the model will run
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Hyperparameters of the model
img_size = 64
batch_size = 64
epochs = 10
n_channels = 3
n_classes = 1
lr = 0.007

# Define the dataset
train_set, val_set, test_set = sets(image_size = img_size)
m_train, m_val, m_test = len(train_set), len(val_set), len(test_set)
input_shape = train_set.shape()

print("Some samples of the dataset.....")
show_samples(train_set, 5)


# Loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle = True)


# Create the model
model = UNET(n_channels, n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
BinaryCrossEntropy = torch.nn.BCEWithLogitsLoss()

print("\nModel summary:")
summary(model, input_shape)

# Training phase

hist_train = []
hist_validation = []

print("Training the model....")
start_time = time.time()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    # Train step
    for idx, (image, mask) in enumerate(train_loader):
        optimizer.zero_grad()
        predicted_mask = model.forward(image)
        loss = BinaryCrossEntropy(predicted_mask, mask)
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    
    train_loss = epoch_loss/m_train
    hist_train.append(train_loss.detach().numpy())

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for image, mask in valid_loader:
            predicted = model.forward(image)
            val_loss += BinaryCrossEntropy(predicted, mask)
        
    val_loss /= m_val
    hist_validation.append(val_loss.detach().numpy())

    print(f"Epoch {epoch + 1} ==> Train loss : {train_loss:.5f} | Validation loss : {val_loss:.5f}")

end_time = time.time() - start_time
hours, minutes, seconds = seconds_to_hhmmss(end_time)
if hours >= 1:
    print(f"Training complete! It took {hours} hours, {minutes} minutes and {seconds} seconds. \n")
elif minutes >= 1:
    print(f"Training complete! It took {minutes} minutes and {seconds} seconds. \n")
else:
    print(f"Training complete! It took {seconds} seconds. \n")

# Saving the model
path = "model/"
name = "UNET_weights.pth"
torch.save(model.state_dict(), path + name)
print(f"UNET weights saved as {name}.")

# Plot losses
plot_losses(hist_train, hist_validation)

# Accuracy
train_acc = check_acc(model, train_loader)
valid_acc = check_acc(model, valid_loader)
print(f"Training set accuracy : {train_acc*100}%")
print(f"Test set accuracy : {valid_acc*100}%")

