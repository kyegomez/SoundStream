import torch
from torch import nn, Tensor

# from zeta import RVQ
class ResidualUnit(nn.Module):
    """
    ResidualUnit is a module that represents a residual unit in a neural network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Example:
    x = torch.randn(1, 128, 100)
    residual_unit = ResidualUnit(128, 128)
    out = residual_unit(x)
    print(out.shape)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        stride: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.stride = stride

        # Conv1d with kernel_size=7 is used to change the number of channels
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            dilation=dilation,
            stride=stride,
            # padding=7,
            *args,
            **kwargs,
        )

        # Conv1d with kernel_size=1 is used to change the number of channels
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=dilation,
            stride=stride,
            *args,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ResidualUnit module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        skip = x
        print(f"input: {x.shape}")
        x = self.conv1(x)
        print(f"conv1: {x.shape}")
        x = self.conv2(x)
        print(f"conv2: {x.shape}")
        return x  # + skip


x = torch.randn(1, 128, 100)
residual_unit = ResidualUnit(128, 128, dilation=1)
out = residual_unit(x)
print(out.shape)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.k = 2 * stride

        self.ru1 = ResidualUnit(in_channels, out_channels, dilation=1)

        self.ru2 = ResidualUnit(
            out_channels, out_channels, dilation=3
        )

        self.ru3 = ResidualUnit(
            out_channels, out_channels, dilation=9
        )

        self.ru4 = ResidualUnit(
            out_channels,
            out_channels,
            kernel_size=self.k,
            stride=self.stride,
        )
