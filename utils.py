# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random, torch, os
import numpy as np


class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import torch.distributed as dist
from collections import defaultdict

class LayeredResidualCollector:
    def __init__(self, device='cuda'):
        self.device = device
        # Use a defaultdict to store states for each layer
        self.good_states_by_layer = defaultdict(list)
        self.bad_states_by_layer = defaultdict(list)

        # Placeholders for gathered states on rank 0
        self.all_good_states_by_layer = None
        self.all_bad_states_by_layer = None

    def add_state(self, hidden_state, is_correct, layer_index):
        """Add a hidden state for a specific layer to the appropriate local collection."""
        # The state for a single layer is typically (batch_size, hidden_dim)
        # We detach and move to CPU to avoid holding GPU memory
        hidden_state = hidden_state.detach().cpu()
        if is_correct:
            self.good_states_by_layer[layer_index].append(hidden_state)
        else:
            self.bad_states_by_layer[layer_index].append(hidden_state)

    def _gather_states(self):
        """Gathers all layer-specific states from all ranks to rank 0."""
        if not dist.is_initialized():
            self.all_good_states_by_layer = self.good_states_by_layer
            self.all_bad_states_by_layer = self.bad_states_by_layer
            return

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if rank == 0:
            good_list_from_all = [None] * world_size
            bad_list_from_all = [None] * world_size
            dist.gather_object(self.good_states_by_layer, good_list_from_all, dst=0)
            dist.gather_object(self.bad_states_by_layer, bad_list_from_all, dst=0)

            # Combine dictionaries from all ranks
            self.all_good_states_by_layer = defaultdict(list)
            for d in good_list_from_all:
                for layer, states in d.items():
                    self.all_good_states_by_layer[layer].extend(states)

            self.all_bad_states_by_layer = defaultdict(list)
            for d in bad_list_from_all:
                for layer, states in d.items():
                    self.all_bad_states_by_layer[layer].extend(states)
        else:
            dist.gather_object(self.good_states_by_layer, None, dst=0)
            dist.gather_object(self.bad_states_by_layer, None, dst=0)

        dist.barrier()

    def compute_residual_for_layer(self, layer_index):
        """
        Computes the residual vector and mean vectors for a specific layer.
        This should be called after _gather_states.
        The computation and result only exist on rank 0.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return None, None, None

        good_states = self.all_good_states_by_layer.get(layer_index, [])
        bad_states = self.all_bad_states_by_layer.get(layer_index, [])

        if len(good_states) < 10 or len(bad_states) < 10:
            print(f"Warning: Insufficient samples for layer {layer_index}. Returning None.")
            return None, None, None

        good_states_tensor = torch.cat(good_states, dim=0).to(self.device)
        bad_states_tensor = torch.cat(bad_states, dim=0).to(self.device)

        good_mean = torch.mean(good_states_tensor, dim=0)
        bad_mean = torch.mean(bad_states_tensor, dim=0)
        residual = good_mean - bad_mean

        return residual, good_mean, bad_mean
