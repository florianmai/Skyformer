"""This module mixes information from different tokens via HyperMixing.
It can be viewed as a linear-time drop-in replacement for (self-)attention.

source: https://arxiv.org/abs/2203.03691

Authors
 * Florian Mai 2023
 * Juan Pablo Zuluaga 2023
"""
import torch
import random, math
import numpy as np
from torch import nn
from typing import Optional

class ConvolutionModule(nn.Module):
    """This is an implementation of convolution module in Conformer.

    Arguments
    ----------
    input_size : int
        The expected size of the input embedding dimension.
    kernel_size: int, optional
        Kernel size of non-bottleneck convolutional layer.
    bias: bool, optional
        Whether to use bias in the non-bottleneck conv layer.
    dropout: float, optional
         Dropout rate.
    dilation: int, optional
         Dilation factor for the non bottleneck conv layer.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = ConvolutionModule(512, 3)
    >>> output = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        config
    ):
        super().__init__()

        input_size = config['transformer_dim']
        kernel_size = config['kernel_size']
        bias = True
        dropout = config['attention_dropout']
        dilation= 1

        self.padding = (kernel_size - 1) * 2 ** (dilation - 1) // 2

        self.layer_norm = nn.LayerNorm(input_size)
        self.bottleneck = nn.Sequential(
            # pointwise
            nn.Conv1d(
                input_size, 2 * input_size, kernel_size=1, stride=1, bias=bias
            ),
            nn.GLU(dim=1),
        )
        # depthwise
        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
            groups=input_size,
            bias=bias,
        )

        self.after_conv = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.GELU(),
            # pointwise
            nn.Linear(input_size, input_size, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        """ Processes the input tensor x and returns the output an output tensor"""
        out = self.layer_norm(x)
        out = out.transpose(1, 2)
        out = self.bottleneck(out)
        out = self.conv(out)

        out = out.transpose(1, 2)
        out = self.after_conv(out)
        if mask is not None:
            out.masked_fill_(mask.unsqueeze(-1), 0.0)
        return out

class HyperConformer(nn.Module):
    """
    Combines HyperMixing and a convolution module. Specifically, first calls the
    convolution module and then the HyperMixing module on top of those outputs.       
    """
    def __init__(
        self, config, global_mixing=None
    ) -> None:
        super().__init__()

        self.conv_module = ConvolutionModule(config)

        if global_mixing is None:
            self.global_module = HyperMixing(config)
        else:
            self.global_module = global_mixing


    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask: Optional[torch.Tensor] = None):

        query = query + self.conv_module(query, key_padding_mask)
        out = self.global_module(query, query, query, key_padding_mask)

        return out

class HyperMixing(nn.Module):
    """ This class implements multi-head HyperMixing.
    It is an implementation of the token-mixing component in HyperMixer, a linear
    time drop-in replacement for self-attention. In contrast to the original HyperMixer,
    this module supports multiple heads, which improves the expressiveness of the model
    while decreasing the number of parameters.

    Reference: https://arxiv.org/abs/2203.03691

    Arguments
    ----------
    input_output_dim : int
        number of features in keys, queries, and values
    hypernet_size : int
        determines the size of the hidden layer of the token-mixing MLP.
    tied : bool
        If True, then the generated weight matrices of the token-mixing MLP are tied.
    num_heads : int
        parallel token-mixing MLPs.
    fix_tm_hidden_size : bool
        If True, the hidden-layer size is equal to hypernet_size rather than hypernet_size / num_heads.
    max_length : int
        Maximum number of input tokens. Needed for generating sufficiently large position embeddings.

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = HyperMixing(512, 2048, num_heads=8)
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """
    def __init__(
        self, config
    ) -> None:

        input_output_dim = config['transformer_dim']
        hypernet_size = config['hypernet_size']
        tied = config['tied_hypernetworks']
        num_heads = config['num_head']
        fix_tm_hidden_size = False
        max_length = config['max_seq_len']

        super().__init__()
        self.input_output_dim = input_output_dim
        self.hyper = _HyperNetwork(input_output_dim, hypernet_size, tied=tied, num_heads=num_heads, keep_output_size=fix_tm_hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(input_output_dim)
        self.num_heads = num_heads

        # add pos encoding
        self.positional_encoding = _PositionalEncoding(
                input_output_dim, max_length
        )

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask: Optional[torch.Tensor] = None):
        """
        The signature of this method is deliberately chosen to be the same as for
        sb.nnet.attention.MultiHeadAttention for compatibility within SpeechBrain.
        
        NOTE: key, value, attn_mask and pos_embs have no effect. Query is used for
        all three. Thus, the module should only be used to replace self-attention at the moment.

        Arguments
        ----------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
            Currently unused. All 
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
            Currently unused.
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.

        Outputs
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        """

        # NOTE: We are ignoring keys and values, because HyperMixing can only be used in the encoder atm (where it's all the same)
        out = query

        bsize = out.size(0)
        seq_len = out.size(1)

        float_mask = torch.logical_not(key_padding_mask).unsqueeze(-1).float()
        out = out * float_mask

        # add position embedding before passing to hypernetwork
        hyp_input = out + self.positional_encoding(out)
        W1, W2 = self.hyper(hyp_input) # [bsize, num_heads, seq_len, hypernet_size // num_heads]

        W1 = W1 * float_mask.unsqueeze(1)
        W2 = W2 * float_mask.unsqueeze(1)

        # reshape the num_heads into the batch dimension for parallelizing
        out = out.transpose(1,2) # [bsize, input_output_dim, seq_len]
        out = out.reshape((bsize * self.num_heads, self.input_output_dim // self.num_heads, seq_len)) # [bsize * num_heads, input_output_dim // num_heads, seq_len]
        W1 = W1.reshape((bsize * self.num_heads, seq_len, -1))
        W2 = W2.reshape((bsize * self.num_heads, seq_len, -1))
        

        # we stick the token-mixing MLP together manually
        out = _mlp_pass_from_components(out, W1, W2, self.activation)

        # concatenate heads
        out = out.reshape((bsize, self.input_output_dim, seq_len))

        # transpose back
        out = out.transpose(1,2)
        
        # apply layer norm on outputs of the TM-MLP
        out = self.layer_norm(out)

        return out

class _HyperNetwork(nn.Module):
    def __init__(
        self, input_output_dim: int, hypernet_size: int, tied=False, num_heads=1, keep_output_size=True
    ) -> None:
        super().__init__()

        self.tied = tied
        self.w1_gen = _ParallelMLPs(input_output_dim, input_output_dim, output_size=hypernet_size, num_mlps=num_heads, keep_output_size=keep_output_size)
        if self.tied:
            self.w2_gen = self.w1_gen
        else:
            self.w2_gen = _ParallelMLPs(input_output_dim, input_output_dim, output_size=hypernet_size, num_mlps=num_heads, keep_output_size=keep_output_size)

    def forward(self, input_tensor: torch.Tensor):
        """
        input_tensor : [batchsize, max_positions, d]
        The HyperNetwork is supposed to generate an MLP of the form W_2(GELU(W1 x)), where
        W1 : N -> k and W2 : k -> N, so it has to return W1 and W2
        """
        W1 = self.w1_gen(input_tensor)
        if self.tied:
            W2 = W1
        else:
            W2 = self.w2_gen(input_tensor)

        return W1, W2

class _ParallelMLPs(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, num_mlps=1, keep_output_size=True):
        super(_ParallelMLPs, self).__init__()

        if output_size is None:
            output_size = input_size

        self.original_in_size = input_size
        self.original_out_size = output_size

        assert input_size % num_mlps == 0
        assert output_size % num_mlps == 0
        assert hidden_size % num_mlps == 0
        input_size = input_size // num_mlps

        if not keep_output_size:
            output_size = output_size // num_mlps
        hidden_size = hidden_size // num_mlps

        self.input_size = input_size
        self.output_size = output_size

        self.num_mlps = num_mlps

        self.fc1_weights = nn.Parameter(torch.empty(num_mlps, hidden_size, input_size))
        self.fc1_biases = nn.Parameter(torch.empty(num_mlps, hidden_size))
        self.fc2_weights = nn.Parameter(torch.empty(num_mlps, output_size, hidden_size))
        self.fc2_biases = nn.Parameter(torch.empty(num_mlps, output_size))

        nn.init.xavier_uniform_(self.fc1_weights, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc1_biases, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc2_weights, gain=math.sqrt(2.0))
        nn.init.xavier_uniform_(self.fc2_biases, gain=math.sqrt(2.0))

        self.activation = nn.GELU()

    def forward(self, x):
        # x [bsize, seq_len, num_features]

        bsize = x.size(0)
        seq_len = x.size(1)

        x = x.reshape((bsize, seq_len, self.num_mlps, self.input_size))
        x = torch.einsum("blmf,mhf->bmlh", x, self.fc1_weights) + self.fc1_biases.unsqueeze(0).unsqueeze(2)
        x = self.activation(x)
        x = torch.einsum("bmlh,mfh->bmlf", x, self.fc2_weights) + self.fc2_biases.unsqueeze(0).unsqueeze(2)

        return x

def _mlp_pass_from_components(out, W1, W2, activation):
    # we stick MLP1 together manually
    out = torch.bmm(out, W1)
    out = activation(out)
    out = torch.bmm(out, W2.transpose(1, 2))
    return out

# NOTE: this is the same implementation as in speechbrain.lobes.models.transformer.Transformer.
# Importing it here leads to a circular dependency. In the future, PositionalEncoding should probably be part of sb.nnet
class _PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = _PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()

