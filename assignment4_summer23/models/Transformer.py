"""
Transformer model.  (c) 2021 Georgia Tech

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
from torch import nn
import random

####### Do not modify these imports.

class TransformerTranslator(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(TransformerTranslator, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q
        
        seed_torch(0)
        
        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        
        print("input size and embedding size ", self.input_size, self.word_embedding_dim)
        self.embeddingL = nn.Embedding(self.input_size, self.word_embedding_dim)      #initialize word embedding layer
        self.posembeddingL = nn.Embedding(self.max_length, self.word_embedding_dim)   #initialize positional embedding layer
        

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        
        ##############################################################################
        # Deliverable 2: Initializations for multi-head self-attention.              #
        # You don't need to do anything here. Do not modify this code.               #
        ##############################################################################
        
        # Head #1
        self.k1 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v1 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q1 = nn.Linear(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = nn.Linear(self.hidden_dim, self.dim_k)
        self.v2 = nn.Linear(self.hidden_dim, self.dim_v)
        self.q2 = nn.Linear(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        
        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the feed-forward layer.        # 
        # Don't forget the layer normalization.                                      #
        ##############################################################################

        self.lin1 = nn.Linear(self.hidden_dim, self.dim_feedforward)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.dim_feedforward, self.hidden_dim)
        self.norm_mh_ff = nn.LayerNorm(self.hidden_dim)
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
        ##############################################################################
        # TODO:
        # Deliverable 4: Initialize what you need for the final layer (1-2 lines).   #
        ##############################################################################
        self.lin_final = nn.Linear(self.hidden_dim, self.output_size)
        self.softmax_final = nn.Softmax(dim=2)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling TransformerTranslator class methods here.      #
        #############################################################################
        outputs = self.final_layer(self.feedforward_layer(self.multi_head_attention(self.embed(inputs))))
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
        #print("input dims, ", inputs.shape)



        N, T = inputs.shape        

        pos = torch.zeros(N,T, dtype = torch.int)

        for i in range(N):
            for j in range(T):
                pos[i,j] = j

        #print("pos dims, ", pos.shape)

        embeddings1 = self.embeddingL(inputs)
        
        
        embeddings2 = self.posembeddingL(pos)

        embeddings = embeddings1 + embeddings2

        #embeddings = embeddings.float().to(self.device)

        print("shape of embeddings", embeddings1.shape, embeddings2.shape)
        #print("type", type(embeddings))
      
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
        outputs = None      #remove this line when you start implementing your code

        K1 = self.k1(inputs)
        Q1 = self.q1(inputs)
        V1 = self.v1(inputs)

        K2 = self.k2(inputs)
        Q2 = self.q2(inputs)
        V2 = self.v2(inputs)

        print("shape of inputs", inputs.shape)
        print("shapes Q1, K1, V1", Q1.shape, K1.shape, V1.shape)

        # Compute head1
        K1t = torch.transpose(K1, 1, 2)

        print("shape of K1 transpose", K1t.shape)

        Q1K1t = torch.bmm(Q1, K1t)

        print("shape of Q1K1t", Q1K1t.shape)

        sqrt_Dk_Reciprocal = 1 / (self.dim_k ** 0.5)

        E = Q1K1t * sqrt_Dk_Reciprocal

        print("shape of E = Q1K1t / torch.sqrt(self.dim_k)", E.shape)

        A1 = self.softmax(E)

        print("shape of A1 = softmax(E)", A1.shape)

        head1 = torch.bmm(A1, V1)

        print("shape of head1 which is A1*V1 ", head1.shape)


        # Compute head2
        K2t = torch.transpose(K2, 1, 2)
        Q2K2t = torch.bmm(Q2, K2t)

        A2 = self.softmax( Q2K2t / self.dim_k**0.5 )

        head2 = torch.bmm(A2, V2)

        print("dims of heads", head1.shape, head2.shape)

        # Concatenate heads
        heads = torch.cat((head1, head2), 2)

        print("shaoe of heads concatenated", heads.shape)


        outputs = self.norm_mh(inputs + self.attention_head_projection(heads))

        print("shape of putputs", outputs.shape)


        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################

        outputs = self.lin1(inputs)
        outputs = self.relu(outputs)
        outputs = self.lin2(outputs)

        outputs = self.norm_mh_ff(inputs + outputs)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        outputs = self.lin_final(inputs)
        #outputs = self.softmax_final(outputs)

        print("shape of outs", outputs.shape)
                
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

class FullTransformerTranslator(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        seed_torch(0)

        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the Transformer Layer          #
        ##############################################################################

        ##############################################################################
        # TODO:
        # Deliverable 2: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Initialize the word embeddings before the positional encodings.            #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################


        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the final layer.               #
        ##############################################################################

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, src, tgt):
        """
         This function computes the full Transformer forward pass used during training.
         Put together all of the layers you've developed in the correct order.

         :param src: a PyTorch tensor of shape (N,T) these are tokenized input sentences
                tgt: a PyTorch tensor of shape (N,T) these are tokenized translations
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the full Transformer stack for the forward pass. #
        #############################################################################
        outputs=None
        # embed src and tgt for processing by transformer

        # create target mask for decoder

        # invoke transformer to generate output

        # pass through final layer to generate outputs

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def generate_translation(self, src):
        """
         This function generates the output of the transformer taking src as its input
         it is assumed that the model is trained. The output would be the translation
         of the input

         :param src: a PyTorch tensor of shape (N,T)

         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 5: You will be calling the transformer forward function to    #
        # generate the translation for the input.                                   #
        #############################################################################
        outputs = None
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True