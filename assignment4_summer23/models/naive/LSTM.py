"""
LSTM model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # An implementation of naive LSTM using Pytorch Linear layers and activations
    # You will need to complete the class init function, forward function and weight initialization

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################

        # i_t: input gate
        self.wii =  nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.bii =  nn.Parameter(torch.Tensor(self.hidden_size))
        

        self.whi =  nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bhi =  nn.Parameter(torch.Tensor(self.hidden_size))

        self.isigmoid = nn.Sigmoid()

        # f_t: the forget gate
        self.wif =  nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.bif =  nn.Parameter(torch.Tensor(self.hidden_size))

        self.whf =  nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bhf =  nn.Parameter(torch.Tensor(self.hidden_size))

        self.fsigmoid = nn.Sigmoid()

        # g_t: the cell gate
        self.wig =  nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.big =  nn.Parameter(torch.Tensor(self.hidden_size))

        self.whg =  nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bhg =  nn.Parameter(torch.Tensor(self.hidden_size))

        self.gtanh = nn.Tanh()

        # o_t: the output gate
        self.wio =  nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.bio =  nn.Parameter(torch.Tensor(self.hidden_size))

        self.who =  nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.bho =  nn.Parameter(torch.Tensor(self.hidden_size))

        self.osigmoid = nn.Sigmoid()

        self.htanh = nn.Tanh()

        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              # 
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################

        # Reference: https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091

        N, seq_sz, _ = x.size()
        
        
        if init_states is None:
            h_t, c_t = (
                torch.zeros(N, self.hidden_size).to(x.device),
                torch.zeros(N, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states
            
        for t in range(seq_sz):
            x_t = x[:, t, :]
            
            i_t = self.isigmoid(x_t @ self.wii + self.bii + h_t @ self.whi + self.bhi)
            f_t = self.fsigmoid(x_t @ self.wif + self.bif + h_t @ self.whf + self.bhf)
            g_t = self.gtanh(x_t @ self.wig + self.big + h_t @ self.whf + self.bhg)
            o_t = self.osigmoid(x_t @ self.wio + self.bio + h_t @ self.who + self.bho)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.htanh(c_t)
            
        
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
