#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:39:37 2017

@author: joni
"""

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable

from ConvLSTMCell import ConvLSTMCell


class GenerativeCell(nn.Module):
    """
    Single generative layer
    """

    def __init__(self, input_size, hidden_size, error_init_size=None):
        """
        Create a generative cell (error, top_down_state, r_state) -> r_state
        :param input_size: {'error': error_size, 'up_state': r_state_size}, r_state_size can be 0
        :param hidden_size: int, shooting dimensionality
        :param error_init_size: tuple, full size of initial (null) error
        """
        super(GenerativeCell,self).__init__()
        self.input_size = input_size
        #print "input_size", self.input_size
        self.hidden_size = hidden_size
        self.error_init_size = error_init_size
        self.memory = ConvLSTMCell(input_size['error']+input_size['up_state'], hidden_size)

    def forward(self, error, top_down_state, state):
        if error is None:  # we just started
            error = Variable(torch.zeros(self.error_init_size))
            
            
        # print error.data.shape
        # model_input = error
        
        #print "error.size()", error.size()
        if top_down_state is not None:
            
            if torch.cuda.is_available() and error.is_cuda == False:
                error = error.cuda()
            if torch.cuda.is_available() and  top_down_state.is_cuda == False:
                top_down_state = top_down_state.cuda()
            #print "top_down_state.data.size()", (top_down_state.data.size())
            model_input = torch.cat((error, f.upsample(top_down_state, scale_factor=2)), 1)
        else:
            model_input = error
        #print top_down_state    
        #print "model_input", model_input.size()
        if torch.cuda.is_available():
            model_input=model_input.cuda()
#            state=state.cuda()
        
        #print "state", state
        return self.memory(model_input, state)

 


def _test_layers():
    state = _test_layer2()
    _test_layer1(top_down_state=state)


if __name__ == '__main__':
    _test_layers()

 