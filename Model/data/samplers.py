# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch

# Used to sample elements randomly from a given list of indices. 
# This is useful in scenarios where you want to create batches of 
# data from a larger dataset, but only want to use a specific subset 
# of that data in a random order.

class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    # __init__: The constructor for the SubsetRandomSampler class.
    
    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    # __iter__: A special method that makes the class iterable. It returns an iterator that 
    # yields elements from 'indices' in a random order each time it's called.

    def __iter__(self):
        # Shuffles the indices using a random permutation and yields elements based on the shuffled order.
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    # __len__: Returns the number of elements in the sampler (length of the indices list).

    def __len__(self):
        return len(self.indices)

    # set_epoch: A method to set the current epoch number. This can be used for tracking or other purposes 
    # in training loops, but doesn't affect the functionality of the sampler directly.

    def set_epoch(self, epoch):
        self.epoch = epoch  # Sets the current epoch number.