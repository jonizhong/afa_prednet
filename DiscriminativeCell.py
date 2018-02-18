#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:38:01 2017

@author: joni
"""

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2
POOL = 2


class DiscriminativeCell(nn.Module):
    """
    Single discriminative layer
    """

    def __init__(self, input_size, hidden_size, first=False):
        """
        Create a discriminative cell (bottom_up, r_state) -> error
        :param input_size: {'input': bottom_up_size, 'state': r_state_size}
        :param hidden_size: int, shooting dimensionality
        :param first: True/False
        """
        super(DiscriminativeCell,self).__init__()
        self.input_size = input_size
        #print "self.input_size", self.input_size
        self.hidden_size = hidden_size
        #print "self.hidden_size", self.hidden_size
        self.first = first
        if not first:
            self.from_bottom = nn.Conv2d(input_size['input'], hidden_size, KERNEL_SIZE, padding=PADDING)
        self.from_state = nn.Conv2d(input_size['state'], hidden_size, KERNEL_SIZE, padding=PADDING)

    def forward(self, bottom_up, state):
        if self.first:
            input_projection = bottom_up
        else:
            input_projection = f.relu(f.max_pool2d(self.from_bottom(bottom_up), POOL, POOL))
            
        # input_projection = self.first and bottom_up  or f.relu(f.max_pool2d(self.from_bottom(bottom_up), POOL, POOL))
        state_projection = f.relu(self.from_state(state))
        
        
        #print state_projection.data.size()
        #print "--"
        #print input_projection.data.size()
        
        #print state_projection[0,0,:,:]
        
        
        
        error = f.relu(torch.cat((input_projection - state_projection, state_projection - input_projection), 1))
        return error


 


def _test_layer2(input_error):
    print('Define model for layer 2')
    discriminator = DiscriminativeCell(input_size={'input': 6, 'state': 32}, hidden_size=32, first=False)

    print('Define a new, smaller state')
    system_state = Variable(torch.randn(1, 32, 4, 6))

    print('Forward layer 1 output and state to the model')
    e = discriminator(input_error, system_state)

    # print output size
    print('Layer 2 error has size', list(e.data.size()))


def _test_layers():
    error = _test_layer1()
    _test_layer2(input_error=error)


if __name__ == '__main__':
    _test_layers()

 