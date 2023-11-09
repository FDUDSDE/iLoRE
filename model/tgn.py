import logging
import random

import numpy as np
import torch
import math
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode, CountEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False, use_state = False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False,
               block_number=6, head_number=3):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    # self.n_node_features = self.node_raw_features.shape[1]
    self.n_node_features = memory_dimension  #todo:check
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep

    self.block_number = block_number
    self.head_number = head_number

    self.use_memory = use_memory
    self.use_state = use_state   #todo: check
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.count_encoder = CountEncode(dimension=self.n_node_features)
    self.memory = None
    self.mini_memory = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(memory=self.memory,
                                               time_encoder=self.time_encoder,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               block_number=self.block_number,
                                               head_number=self.head_number,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 count_encoder=self.count_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 n_count_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20, mini_batch_size=50, seed = 0):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    num_mini_batch = math.ceil(n_samples / mini_batch_size)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    node_to_update = []
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        # message
        mini_memory, last_update, node_to_update, trans_memory, time_record = \
                                    self.get_updated_mini_memory(list(range(self.n_nodes)), self.memory.message_list, seed)
        memory = self.get_updated_memory(node_to_update, trans_memory, time_record, mini_memory, seed)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if self.use_memory:
      if self.memory_update_at_start:
        trans_memory, node_to_update_pos, time_record_pos = self.update_mini_memory(positives, seed)
        self.update_memory(node_to_update_pos, trans_memory, time_record_pos, seed)

      for mini_batch_idx in range(0, num_mini_batch):
        if mini_batch_idx >= num_mini_batch:
            continue
        mini_start_idx = mini_batch_idx * mini_batch_size
        mini_end_idx = min(n_samples, mini_start_idx + mini_batch_size)
        mini_sources_batch, mini_destinations_batch = source_nodes[mini_start_idx:mini_end_idx], \
                destination_nodes[mini_start_idx:mini_end_idx]
        mini_edge_idxs_batch = edge_idxs[mini_start_idx:mini_end_idx]
        mini_timestamps_batch = edge_times[mini_start_idx:mini_end_idx]

        unique_sources, source_id_to_messages = self.get_raw_messages(mini_sources_batch,
                                                                      None,
                                                                      mini_destinations_batch,
                                                                      None,
                                                                      mini_timestamps_batch, mini_edge_idxs_batch)
        unique_destinations, destination_id_to_messages = self.get_raw_messages(mini_destinations_batch,
                                                                                None,
                                                                                mini_sources_batch,
                                                                                None,
                                                                                mini_timestamps_batch, mini_edge_idxs_batch)

        self.memory.store_raw_messages(unique_sources, source_id_to_messages, mini_batch_idx)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages, mini_batch_idx)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]

    return source_node_embedding, destination_node_embedding, negative_node_embedding, memory[positives], self.memory.get_memory(positives)

  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, seed, n_neighbors=20, mini_batch_size=50):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding, memory_pos1, memory_pos2 = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors, mini_batch_size, seed)

    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=0)
    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid(), memory_pos1, memory_pos2

  def update_memory(self, node_to_update, trans_memory, time_record, seed):
    if len(self.memory.message_list) <= 0:
        return

    node_to_update = np.unique(node_to_update)
    trans_memory = trans_memory[node_to_update, :, :]
    time_record = time_record[node_to_update, :]

    self.memory_updater.update_memory(node_to_update, trans_memory,time_record, seed)


  def update_mini_memory(self, nodes, seed):
    message_list = self.memory.message_list
    node_to_update = []
    trans_memory = torch.zeros((self.n_nodes, len(message_list), self.memory_dimension), requires_grad=False).to(self.device)
    time_record = torch.zeros((self.n_nodes, len(message_list)), requires_grad=False).to(self.device)
    for messages_idx in range(len(message_list)):
        messages = message_list[messages_idx]
        unique_nodes, unique_messages, unique_timestamps = \
          self.message_aggregator.aggregate(
            nodes,
            messages)

        if len(unique_nodes) > 0:
          unique_messages = self.message_function.compute_message(unique_messages)

        self.memory_updater.update_mini_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps, seed=seed)
        node_to_update.extend(unique_nodes)
        trans_memory[:, messages_idx, :] = self.memory.memory
        time_record[unique_nodes, messages_idx] = self.memory.last_update[unique_nodes]

        self.memory.clear_messages(messages_idx, unique_nodes)

    node_to_update = np.unique(node_to_update)

    return trans_memory, node_to_update, time_record



  def get_updated_mini_memory(self, nodes, message_list, seed):
    # Aggregate messages for the same nodes
    node_to_update = []
    trans_memory = torch.zeros((self.n_nodes, len(message_list), self.memory_dimension), requires_grad=False).to(self.device)
    time_record = torch.zeros((self.n_nodes, len(message_list)), requires_grad=False).to(self.device)
    # mini_memory = self.memory.mini_memory.data.clone()
    mini_memory = self.memory.memory.data.clone()
    last_update = self.memory.last_update.data.clone()
    for message_idx in range(len(message_list)):
        messages = message_list[message_idx]
        unique_nodes, unique_messages, unique_timestamps = \
          self.message_aggregator.aggregate(
            nodes,
            messages)

        if len(unique_nodes) > 0:
          unique_messages = self.message_function.compute_message(unique_messages)

        mini_memory, last_update = self.memory_updater.get_updated_mini_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 mini_memory,
                                                                                 last_update,
                                                                                 timestamps=unique_timestamps,
                                                                               seed=seed)
        trans_memory[:, message_idx, :] = mini_memory
        time_record[unique_nodes, message_idx] = last_update[unique_nodes]
        node_to_update.extend(unique_nodes)

    node_to_update = np.unique(node_to_update)

    return mini_memory, last_update, node_to_update, trans_memory, time_record

  def get_updated_memory(self, nodes, trans_memory, time_record, memory, seed):

      trans_memory = trans_memory[nodes, :, :]
      time_record = time_record[nodes, :]
      updated_memory = self.memory_updater.get_updated_memory(nodes, trans_memory, time_record, memory, seed)

      return updated_memory


  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)

    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
