import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    self.__init_memory__()

  def __init_last__(self):
    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)

    # self.messages = defaultdict(list)
    self.message_list = list()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)

    # Mini_Memory
    self.mini_memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),  #todo
                                    requires_grad=False)

    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)

    # self.messages = defaultdict(list)
    self.message_list = list()

  def store_raw_messages(self, nodes, node_id_to_messages, mini_batch_idx):
    if mini_batch_idx >= len(self.message_list):
      self.message_list.append(defaultdict(list))
    for node in nodes:
      self.message_list[mini_batch_idx][node].extend(node_id_to_messages[node])

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :]


  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values


  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_memory(self):
    message_list_clone = list()

    for messages in self.message_list:
      messages_clone = defaultdict(list)
      for k, v in messages.items():
        messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]
      message_list_clone.append(messages_clone)

    return self.mini_memory.data.clone(), self.memory.data.clone(), self.last_update.data.clone(), message_list_clone


  def restore_memory(self, memory_backup):
    self.mini_memory.data, self.memory.data, self.last_update.data = \
      memory_backup[0].clone(), memory_backup[1].clone(), memory_backup[2].clone()

    self.message_list = list()
    for messages in memory_backup[3]:
      messages_clone = defaultdict(list)
      for k, v in messages.items():
        messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]
      self.message_list.append(messages_clone)


  def detach_memory(self):
    self.memory.detach_()
    self.mini_memory.detach_()

    # Detach all stored messages
    for messages_idx in range(len(self.message_list)):
      for k, v in self.message_list[messages_idx].items():
        new_node_messages = []
        for message in v:
          new_node_messages.append((message[0].detach(), message[1]))
        self.message_list[messages_idx][k] = new_node_messages

  def clear_messages(self, message_idx, nodes):
    # self.message_list[message_idx] = defaultdict(list)
    for node in nodes:
      self.message_list[message_idx][node] = []

