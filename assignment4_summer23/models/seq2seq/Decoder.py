"""
S2S Decoder model.  (c) 2021 Georgia Tech

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

import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        self.embedding = nn.Embedding(self.output_size, self.emb_size)
        if model_type == "RNN":
            self.rnn = nn.RNN(emb_size, decoder_hidden_size, batch_first = True) 
        elif model_type == "LSTM":
            self.lstm = nn.LSTM(emb_size, decoder_hidden_size, batch_first = True)
        else:
            print("wrong recurrent layer")

        self.Linear1 = nn.Linear(decoder_hidden_size, output_size)

        self.LogSoftmax = nn.LogSoftmax()

        self.dropout = nn.Dropout(dropout)

        #others
        self.softmaxOnDim1 = nn.Softmax(dim=1)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim) This is K
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """

        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # Summer Session Only: you may use pythorch built-in cosine similarity fn   #
        # DO NOT USE nn.torch.functional.cosine_similarity or some other library    #
        # function. Implement from formula given in docstring directly              #
        #############################################################################


        """
        print("shape of q before reordering", hidden.shape)
        print("hidden", hidden)
        print("shape of encoder outputs or k", encoder_outputs.shape)
        print("encoder_outputs", encoder_outputs)
        """
        q = hidden.squeeze(0).unsqueeze(-2)
        #print("shape of q after reordering", q.shape)
        k = encoder_outputs

        #print("hidden q after fixing dims", q)
        
        cos = nn.CosineSimilarity(dim=2)
        attention = cos(q, k)
        #print("attention before softmax", attention)
        attention = self.softmaxOnDim1(attention).unsqueeze(1)
        #print("attention after softmax", attention)
        #print("shape of attention", attention.shape)



        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention

    def forward(self, input, hidden, encoder_outputs=None, attention=False):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden weights of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the weights coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
            where N is the batch size, T is the sequence length
        """

        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       if attention is true, compute the attention probabilities and use   #
        #       it to do a weighted average on the hidden and cell states to        #
        #       determine what will be consumed by the recurrent layer              #
        #                                                                           #
        #       Apply linear layer and log-softmax activation to output tensor      #
        #       before returning it.                                                #
        #############################################################################

        input_to_rec = self.dropout(self.embedding(input))  # input is (N, 1)
        cell = None

        #print("attention", attention)

        if attention is True:

            if self.model_type == "RNN":
                att_probs = self.compute_attention(hidden, encoder_outputs)

                #att_probs is (dimension: N,1,T); hidden (dimensions: 1,N, hidden_dim); encoder_outputs: (N,T,encoder_hidden_size)
                hidden = torch.bmm(att_probs, encoder_outputs)
                hidden = torch.swapaxes(hidden, 0, 1)
                #print("attention probs * encoder_outputs", hidden)

            else: # LSTM case
                hidden, cell = hidden

                att_probs = self.compute_attention(hidden, encoder_outputs)
                hidden = torch.swapaxes(torch.bmm(att_probs, encoder_outputs), 0, 1)

                att_probs = self.compute_attention(cell, encoder_outputs)
                cell = torch.swapaxes(torch.bmm(att_probs, encoder_outputs), 0, 1)
                




            

        if self.model_type == "RNN":
            #print("hidden dims going into RNN", hidden.shape)
            #print("hidden going into RNN", hidden)
            #print("input going into RNN shape", input_to_rec.shape)
            output, hidden = self.rnn(input_to_rec, hidden)


        elif self.model_type == "LSTM":
            if cell is None:
                (hidden, cell) = hidden
            output, (hidden, cell) = self.lstm(input_to_rec, (hidden, cell))
            hidden = (hidden, cell)
            



        output = output.squeeze(1)

        output = self.LogSoftmax(self.Linear1(output))

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #print("dims of output and hidden", output.shape, hidden.shape)
        return output, hidden
    

