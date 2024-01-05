import torch
from torch import nn, Tensor


class FiLM(nn.Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Linear(dim_cond, dim * 2)

    def forward(self, x: Tensor, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim=-1)
        return x * gamma + beta


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, dilations=[1, 3, 9]
    ):
        """
        Encoder block that consists of a sequence of residual units with dilated convolutions,
        followed by a down-sampling strided convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels after down-sampling.
            stride (int): Stride size for the down-sampling convolution.
            dilations (list of int): Dilation rates for the dilated convolutions.
        """
        super(EncoderBlock, self).__init__()

        # Create the dilated residual units
        self.residual_units = nn.ModuleList(
            [
                self._create_residual_unit(in_channels, dilation)
                for dilation in dilations
            ]
        )

        # Create the down-sampling layer
        self.downsample = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        # activation
        self.act = nn.ELU()

    def forward(self, x):
        # Apply each residual unit in sequence
        for unit in self.residual_units:
            x = unit(x)

        # Down-sample the result
        x = self.downsample(x)
        return x

    def _create_residual_unit(self, channels, dilation):
        """
        Create a residual unit with dilated convolution and ELU activation.

        Args:
            channels (int): Number of channels for the convolution.
            dilation (int): Dilation rate for the convolution.

        Returns:
            nn.Sequential: The residual unit as a sequential model.
        """
        # Calculate padding for causal convolution
        padding = (dilation * (3 - 1), 0)  # kernel size is 3

        return nn.Sequential(
            # Apply padding only to the left side for causal convolution
            nn.ConstantPad1d(padding, 0),
            nn.Conv1d(
                channels, channels, kernel_size=3, dilation=dilation
            ),
            nn.ELU(),
        )


# Example usage
in_channels = 64  # Number of input channels
out_channels = 128  # Number of output channels
stride = 2  # Stride for down-sampling
encoder_block = EncoderBlock(in_channels, out_channels, stride)

# Input tensor with batch size of 1, in_channels, and sequence length of 1024
input_tensor = torch.rand(1, in_channels, 1024)
output = encoder_block(input_tensor)

print(
    output
)  # Output shape will be (Batch Size, out_channels, Sequence Length // stride)
