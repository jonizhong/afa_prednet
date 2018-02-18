#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 17:47:21 2017

@author: joni
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import operator

from torch.autograd import Variable

def align(x, y, start_dim=2):
    xd, yd = x.dim(), y.dim()
    if xd > yd:
        for i in range(xd - yd): y = y.unsqueeze(0)
    elif yd > xd:
        for i in range(yd - xd): x = x.unsqueeze(0)
    xs = list(x.size())
    ys = list(y.size())
    
    print xs, ys
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd-i-1
        if   ys[td]==1: ys[td] = xs[td]
        elif xs[td]==1: xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)

def dot(x, y):
    x, y = align(x, y)
    assert(1<y.dim()<5)
    if y.dim() == 2:
        return x.mm(y)
    elif y.dim() == 3: 
        return x.bmm(y)
    else:
        xs,ys = x.size(), y.size()
        res = torch.zeros(*(xs[:-1] + (ys[-1],)))
        for i in range(xs[0]): res[i] = x[i].bmm(y[i])
        return res

def aligned_op(x,y,f):
    x, y = align(x,y,0)
    return f(x, y)

def add(x, y): return aligned_op(x, y, operator.add)
def sub(x, y): return aligned_op(x, y, operator.sub)
def mul(x, y): 
    

    
    return aligned_op(x, y, operator.mul)
def div(x, y): return aligned_op(x, y, operator.truediv)



class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i - max_i x_i) / sum_j exp(x_j - max_i x_i) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self,  hidden_size):
        super(Attention, self).__init__()
        #self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None
        self.hidden_size = hidden_size
        

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask


    def forward(self, state, action):
        
        # action size: (batch, out_len, in_len)
        
        
        #batch_size = output.size(0) #output size (batch, out_len, image_size)
        
        
        #hidden_size = output.size(2)
        # input_size = action.size(1) #action size (batch, in_len, out_len)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        # attn = torch.bmm(output, context.transpose(1, 2))
        #if self.mask is not None:
        #    attn.data.masked_fill_(self.mask, -float('inf'))
        #attn = F.softmax(attn.view(-1, input_size)).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        
        #print "action.data.size()", action.data.size()
        #print "output.data.size()", output.size()
        
        #TODO: tensor multiplication 
        
        #print "action.data.size()", action.data.size()
        #print "output1.data.size()", output1.data.size()
        #print "output2.data.size()", output2.data.size()
        
        
        input_size = action.size()[1]
        num_classes = len(state)
        
        
        
        self.fc1 = nn.Linear(input_size, self.hidden_size) 
        
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, num_classes)  
        
        
        if torch.cuda.is_available():
            self.fc1 = self.fc1.cuda()
            self.relu = self.relu.cuda()
            self.fc2 = self.fc2.cuda()
        
        # mix = action[0,0] * output1 + action[0,1]*output2
        #action = action.cpu()
        
        out = self.fc1(action)
        out = self.relu(out)
        out = self.fc2(out)
        
        
        # mix = action[0,0].data.numpy() * output1.data.numpy() + action[0,1].data.numpy() * output2.data.numpy()
        mix = 0.0
        for i in range(len(state)):
            mix += state[i]*out[0,i].expand_as(state[i])
        
        
        #mix = torch.unsqueeze(mix,0)
        
         
        return mix
