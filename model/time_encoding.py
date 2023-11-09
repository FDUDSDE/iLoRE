import torch
import numpy as np


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output

class CountEncode(torch.nn.Module):
  # Count Encoding
  def __init__(self, dimension, drop=0.3):
    super(CountEncode, self).__init__()

    self.dimension = dimension
    self.neighbor_co_occurrence_encode_layer = torch.nn.Sequential(
      torch.nn.Linear(in_features=1, out_features=self.dimension),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=drop, inplace=False),
      torch.nn.Linear(in_features=self.dimension, out_features=self.dimension),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=drop, inplace=False),
      torch.nn.Linear(in_features=self.dimension, out_features=self.dimension)
    )

  def forward(self, count):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    count = count.unsqueeze(dim=-1)

    # output has shape [batch_size, seq_len, dimension]
    output = self.neighbor_co_occurrence_encode_layer(count)

    return output
