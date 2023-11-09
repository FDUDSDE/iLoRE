import random

import numpy as np
from torch import nn
import torch
from modules.transformer_encoder import Transformer


class MemoryUpdater(nn.Module):
  def update_mini_memory(self, unique_node_ids, unique_messages, timestamps, seed):
    pass

  def update_memory(self, unique_node_ids, trans_memory, time_record, seed):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, time_encoder, message_dimension, memory_dimension, block_number, head_number, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, trans_memory, time_record, seed):
    if len(unique_node_ids) <= 0:
      return

    updated_memory = self.memory_updater(trans_memory, time_record, seed)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def update_mini_memory(self, unique_node_ids, unique_messages, timestamps, seed):
    if len(unique_node_ids) <= 0:
      return

    mini_memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    updated_mini_memory = self.mini_memory_updater(unique_messages, mini_memory)

    self.memory.set_memory(unique_node_ids, updated_mini_memory)  # set memory


  def get_updated_mini_memory(self, unique_node_ids, unique_messages, mini_memory, last_updated, timestamps, seed):
    if len(unique_node_ids) <= 0:
      return mini_memory.data.clone(), last_updated.data.clone()

    mini_memory[unique_node_ids] = self.mini_memory_updater(unique_messages, mini_memory[unique_node_ids])

    last_updated[unique_node_ids] = timestamps

    return mini_memory, last_updated

  def get_updated_memory(self, unique_node_ids, trans_memory, time_record, memory, seed):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone()

    memory[unique_node_ids] = self.memory_updater(trans_memory, time_record, seed)

    return memory


class TransMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, time_encoder, message_dimension, memory_dimension, block_number, head_number, device):
    super(TransMemoryUpdater, self).__init__(memory, time_encoder,message_dimension, memory_dimension,
                                             block_number, head_number, device)

    self.memory_updater = Transformer(hidden_dim=memory_dimension,
                                      block_num=block_number,
                                      head_num=head_number,
                                      device=device,
                                      time_encoder=time_encoder).to(device)

    self.mini_memory_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


def get_memory_updater(memory, time_encoder, message_dimension, memory_dimension, block_number, head_number, device):
    return TransMemoryUpdater(memory, time_encoder, message_dimension, memory_dimension,
                              block_number, head_number, device)
