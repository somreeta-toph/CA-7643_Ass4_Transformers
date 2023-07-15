import random

""" 			  		 			     			  	   		   	  			  	
Seq2Seq model.  (c) 2021 Georgia Tech

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

import torch
import torch.nn as nn
import torch.optim as optim


# import custom models


class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device, attention=False):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.attention=attention  #if True attention is implemented
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, source, out_seq_len=None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        N = source.shape[0]
        if out_seq_len is None:
            out_seq_len = source.shape[1]
        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden weights being fed into the decoder   #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################

        # Ref: https://www.guru99.com/seq2seq-model.html

        outputs = torch.zeros(N, out_seq_len, self.decoder.output_size)
        #print("dim of source is", source.shape)
        encoder_output, hidden_encoder = self.encoder.forward(source)   # source is of dim (N, T)

        #print("dim of encoder hidden outs", hidden_encoder.shape)    # encoder hidden outs are of dim (1, N, hid_dim)

        hidden = hidden_encoder # initialize the hidden_decoder vector to be the last hidden encoder vector ; (1, N, hid_dim) For the test case, this is 1, 2, 32
         
        decoder_input = source[:, 0].unsqueeze(1) # first input of each of the N source tokens contain the <SOS> dim (N, 1)
        for i in range(out_seq_len):
            
            output, hidden = self.decoder.forward(decoder_input, hidden, encoder_output, self.attention) #input dim (N, 1), hidden_decoder dim (1, N, hid_dim), encoder_outputs dim (N,T,hid_dim), attention=False
            #print("hidden size", hidden.shape)
            # output is (N, output_size)
            # hidden is (1, N, hid_dim)

            #update decoder inputs for the next sequence
            decoder_input = torch.argmax(output, dim=1).unsqueeze(-1) # the one with the max probability is the next input

            #update the outputs
            #output = output.unsqueeze(1)
            #print("shape of output getting updated in outputs and outputs", output.shape, outputs.shape)
            #print("output and outputs", output, outputs)
            #outputs[:, i, :] = output

            for n in range(N):
                for os in range(self.decoder.output_size):
                    outputs[n, i, os] = output[n,os]
            




            


        

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs
    
    
