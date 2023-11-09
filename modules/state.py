import torch
from torch import nn
from torch.autograd import Function
from copy import deepcopy

class State(nn.Module):

    def __init__(self, n_nodes, memory_dimension, device="cpu"):
        super(State, self).__init__()
        self.n_nodes = n_nodes
        self.memory_dimension = memory_dimension
        self.device = device

        self.__init_state__()

    def __init_state__(self):
        """
        Initializes the state to all ones. It should be called at the start of each epoch.
        :return:
        """

        self.state = nn.Parameter(torch.ones((self.n_nodes, 1)).to(self.device),
                                  requires_grad=True)

    def get_state(self, node_idxs):
        return self.state[node_idxs]

    def set_state(self, node_idxs, values):
        # print(node_idxs.size())
        with torch.no_grad():
            self.state[node_idxs] = values

    def detach_state(self):
        self.state.detach_()

    def backup_state(self):
        return self.state.data.clone()

    def restore_state(self, state_backup):
        self.state.data = state_backup.clone()

class StateUpdater(nn.Module):
    def __init__(self, state, memory, memory_dimension, margin, device):
        super(StateUpdater, self).__init__()
        self.state = state
        self.memory = memory
        self.memory_dimension = memory_dimension
        self.margin = margin
        self.device = device

        # 定义线性层
        self.linear = nn.Linear(self.memory_dimension, 1, bias=True).to(self.device)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))


    def update_state(self, unique_node_ids):
        if len(unique_node_ids) <= 0:
            return

        state = self.state.get_state(unique_node_ids)
        memory = self.memory.get_memory(unique_node_ids)

        updated_state, is_updated = self.state_updater(memory, state)

        self.state.set_state(unique_node_ids, updated_state)

        return is_updated, updated_state


    def state_updater(self, memory, state):
        # Compute state delta
        new_state_delta = torch.sigmoid(self.linear(memory))
        # bernoulli
        is_updated = BinaryLayer.apply(state)
        # Compute new state
        new_state = is_updated * new_state_delta + (1. - is_updated) * (state + torch.min(new_state_delta, 1. - state))

        return new_state, is_updated

    def set_updated_state(self, unique_nodes_ids):
        if len(unique_nodes_ids) <= 0:
            return  self.state.state.data.clone()

        updated_state = self.state.state.data.clone()
        memory = self.memory.memory.data.clone()
        updated_state[unique_nodes_ids] = self.state_updater(memory[unique_nodes_ids], updated_state[unique_nodes_ids])

        self.state.set_state(unique_nodes_ids, updated_state)

