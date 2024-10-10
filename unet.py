import torch as torch
import torchvision as tv
from torchsummary import summary


class DoubleConv2D(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DoubleConv2D, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias = False), # Same conv: p = (f-1)/2 = 1 
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias = False), # Same conv: p = (f-1)/2 = 1 
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)


initial_unet_channels = 8

class UNET(torch.nn.Module):    
    def __init__(self, input_channels, num_classes, channels = [initial_unet_channels, initial_unet_channels * 2, initial_unet_channels * 4, initial_unet_channels * 8]):
        super(UNET, self).__init__()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.last_conv = torch.nn.Conv2d(channels[0], num_classes, 1, 1)
        self.downs = torch.nn.ModuleList()
        self.up_tconvs = torch.nn.ModuleList() # Transposed convolutions
        self.up_dconvs = torch.nn.ModuleList() # Double convolutions

        # Encoder
        for channel in channels:
            self.downs.append(DoubleConv2D(input_channels, channel))
            input_channels = channel
        
        # Bottleneck
        self.bottleneck = DoubleConv2D(channels[-1], 2 * channels[-1])

        # Decoder
        for channel in reversed(channels):
            self.up_tconvs.append(torch.nn.ConvTranspose2d(2*channel, channel, 2, 2))
            self.up_dconvs.append(DoubleConv2D(2 * channel, channel))
    
    def forward(self, x):
        skip_cons = []

        # Decoding
        for down in self.downs:
            x = down(x)
            skip_cons.append(x)
            x = self.pool(x)
        
        # Pass through bottle neck
        x = self.bottleneck(x)
        skip_cons = skip_cons[::-1] # Reverse the list 

        # Encoding
        for i in range(len(self.up_tconvs)):
            x = self.up_tconvs[i](x)
            skip_con = skip_cons[i]

            if x.shape != skip_con.shape:
                skip_con = tv.transforms.functional.resize(skip_con, size = x.shape[2:]) # only resize h and w


            x = torch.cat([x, skip_con], dim = 1) # x.shape = (n_batch, c, h, w)
            x = self.up_dconvs[i](x)

        # Final convolution
        x = self.last_conv(x)
        return x
    
if __name__ == "__main__":
    x = torch.rand([3, 64, 64])
    model = UNET(3, 1)
    summary(model, x.shape)
