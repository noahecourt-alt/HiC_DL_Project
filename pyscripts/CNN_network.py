import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CNN(nn.Module): #input is a 4 dim tensor. shape = (batch size, input channels, H, W)
    def __init__(
            self, in_channels: int, channels: list[int] = [16, 32, 16],
            kernel_sizes: list[int] = [7, 5, 3], unet=False, depth=3, base_width=32, diagonal=False #depth is the number of down/upsampling blocks
    ):
    
        super().__init__()

        self.diagonal = diagonal

        #if len(channels) != len(kernel_sizes):
        #    raise ValueError("Channels and Kernel Sizes must be of same length")
        self.kernel_sizes = kernel_sizes
        self.layerseq = []
        self.padding = [(k - 1)//2 for k in self.kernel_sizes] #integer divison so no type error
        channel = in_channels
        self.unet = unet

        if not unet: #ie just a simple CNN
            for c, k, p in zip(channels, kernel_sizes, self.padding):
                self.layerseq.append(nn.Conv2d(channel, c, kernel_size=k, padding=p))
                channel = c
                self.layerseq.append(nn.ReLU())
            self.layerseq.append(nn.Conv2d(channel, 1, kernel_size=1))
            self.net = nn.Sequential(*self.layerseq)

        else: #ie a UNet CNN
            output_feat = base_width
            input_feat = channel
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2) #kernel size and stride chosen to half the H x W
            self.encoders = nn.ModuleList()
            self.decoders = nn.ModuleList()
            self.upsamples = nn.ModuleList()

            encoder_channels = []

            for i in range(depth):
                self.encoders.append(self.encoder_block(input_feat=input_feat, output_feat=output_feat))
                encoder_channels.append(output_feat)
                input_feat = output_feat
                output_feat *= 2

            self.bottleneck = self.encoder_block(input_feat=input_feat, output_feat=output_feat)

            for i in reversed(encoder_channels):
                self.upsamples.append(nn.ConvTranspose2d(output_feat, i, kernel_size=2, stride=2)) #kernel_size and stride chosen to double the H x W
                self.decoders.append(self.encoder_block(input_feat=i*2, output_feat=i))
                output_feat = i

            self.output = nn.Conv2d(output_feat, 1, kernel_size=1)

    def encoder_block(self, input_feat, output_feat):
        channel_out_list = []
        mult = []

        base = output_feat // len(self.kernel_sizes)
        remainder = output_feat % len(self.kernel_sizes)

        mult = [base] * len(self.kernel_sizes)
        for i in range(remainder):
            mult[i] += 1

        for i, (k, m) in enumerate(zip(self.kernel_sizes, mult)):
            for _ in range(m):
                channel_out_list.append(nn.Conv2d(input_feat, 1, kernel_size=k, padding=self.padding[i]))
        
        return nn.ModuleList(channel_out_list) #this is the list that contains a conv layer for each kernel

    def forward(self, x):
        if not self.unet:
            return self.net(x)

        features = [] #features for concatinating later
        for block in self.encoders:
            outputs = []
            for conv in block:
                outputs.append(torch.relu(conv(x)))
            x = torch.cat(outputs, dim=1)
            features.append(x)
            x = self.pool(x)

        outputs = []
        for conv in self.bottleneck:
            outputs.append(torch.relu(conv(x)))
        x = torch.cat(outputs, dim=1)


        for up, dec, f in zip(self.upsamples, self.decoders, reversed(features)):
            x = up(x)
            x = torch.cat((x, f), dim=1)
            outputs = []
            for conv in dec:
                outputs.append(torch.relu(conv(x)))
            x = torch.cat(outputs, dim=1)

        output = self.output(x)
        if self.diagonal:
            return 0.5 * (output + output.transpose(-1, -2)) #enforce symmetry.
        return output

def get_model(model_name: str, in_channels: int, kernel_sizes: list, diagonal: bool, depth: int = 3, base_width: int = 32):
    if model_name == "CNN":
        model = CNN(in_channels=in_channels, kernel_sizes=kernel_sizes, diagonal=diagonal)
    elif model_name == "UNet_CNN":
        model = CNN(in_channels=in_channels, unet=True, kernel_sizes=kernel_sizes, depth=depth, base_width=base_width, diagonal=diagonal)
    elif model_name == "UNet_CNN_smp":
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=in_channels, classes=1, activation=None)
    return model
