import torch
from torch import nn


class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation):
        super(ResidualUnit, self).__init__()
        # The kernel size for the dilated convolution is 7.
        # Padding is adjusted for causality: only applied to the left (past).
        padding = (7 - 1) * dilation // 2
        self.dilated_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=padding,
            dilation=dilation,
        )
        self.activation = nn.ELU()

    def forward(self, x):
        # Apply the dilated convolution followed by an activation function
        conv_out = self.dilated_conv(x)
        activated = self.activation(conv_out)
        # Element-wise sum for the residual connection
        return x + activated


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(EncoderBlock, self).__init__()
        # EncoderBlock contains three ResidualUnit modules with different dilation rates
        self.residual_units = nn.Sequential(
            ResidualUnit(in_channels, dilation=1),
            ResidualUnit(in_channels, dilation=3),
            ResidualUnit(in_channels, dilation=9),
        )
        # Strided convolution for downsampling
        self.downsample = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

    def forward(self, x):
        # Pass through the residual units
        x = self.residual_units(x)
        # Downsample the output
        x = self.downsample(x)
        return x


class Encoder(nn.Module):
    def __init__(self, Cenc, Benc, D, strides):
        super(Encoder, self).__init__()
        # Initial convolution layer to set the number of channels
        self.initial_conv = nn.Conv1d(
            1, Cenc, kernel_size=7, stride=1, padding=3
        )
        # Encoder blocks with increasing channels and varying stride for downsampling
        self.encoder_blocks = nn.ModuleList()
        channels = Cenc
        for i, stride in enumerate(strides):
            self.encoder_blocks.append(
                EncoderBlock(channels, channels * 2, stride)
            )
            channels *= 2
        # Final convolution layer to set the dimensionality of the embeddings to D
        self.final_conv = nn.Conv1d(
            channels, D, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        # Pass the input through the initial convolution
        x = self.initial_conv(x)
        # Pass the result through each encoder block
        for block in self.encoder_blocks:
            x = block(x)
        # Final convolution to obtain the embeddings
        x = self.final_conv(x)
        return x


# Example usage:
Cenc = 64  # Number of channels for the initial convolution
Benc = 4  # Number of encoder blocks
D = 128  # Dimensionality for embeddings
strides = [2, 4, 5, 8]  # Strides for each encoder block

encoder = Encoder(Cenc, Benc, D, strides)
input_tensor = torch.rand(
    1, 1, 1024
)  # Example input tensor representing a single 1D signal
output = encoder(input_tensor)

print(
    output.shape
)  # Output shape will be (Batch Size, D, Compressed Sequence Length)
