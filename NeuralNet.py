import torch.nn.functional as F
from torch import nn

n_in, n_h, n_out, batch_size = 10, 5, 1, 10 #layer size and batch size


x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])