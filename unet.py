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


class UNET(torch.nn.Module):
    def __init__(self, input_channels, num_classes, channels = [64//2**3, 128//2**3, 256//2**3, 512//2**3]):
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
            self.up_tconvs.append(torch.nn.ConvTranspose2d(2*channel, 2*channel, 2, 2))
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
                x = tv.transforms.functional.resize(x, size = skip_con.shape[2:]) # only resize h and w

            concatenated = torch.cat([x, skip_con], dim = 1) # x.shape = (n_batch, c, h, w)
            x = self.up_dconvs[i](x)

        # Final convolution
        x = self.last_conv(x)
        return x