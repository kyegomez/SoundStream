import torch 
from soundstream.model import ResidualUnit


x = torch.randn(1, 128, 100)
residual_unit = ResidualUnit(128, 128, dilation=1)
out = residual_unit(x)
print(out)