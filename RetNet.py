import torch
from torch import nn

class MSR(nn.Module):
    """ Multi-Scale Retention Module """
    def __init__(self, h: int, d_model: int):
        """
        Args:
            h (int): Number of heads.
            d_model (int): Embedding dimension size (the "hidden dimension").
        """
        super().__init__()

        # First line of Equation 8 from the paper
        gammas = 1 - torch.exp2(-5 - torch.arange(0, h))

        # Initialize second line of Equation 8 from the paper
        self.heads = Retention(gammas)

        # Initialize fourth line of Equation 8 from the paper
        self.W_G = nn.Parameter(torch.randn(d_model, d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_model))
        self.swish = nn.SiLU()

    def forward(self, X: torch.Tensor):
        # Third and fourth lines of Equation 8 from the paper
        # NOTE: Move GroupNorm into Retention due to dimensions. Note that the
        # shape of Y is (batch, seq_len, d_model)
        Y = self.heads(X)
        return torch.matmul(self.swish(torch.matmul(X, self.W_G)) * Y, self.W_O)


class FFN(nn.Module):
    """ Feed Forward Network """
    def __init__(self, seq_len: int, d_model: int, FFN_size: int):
        """
        Args:
            seq_len (int): Size of context window.
            d_model (int): Embedding dimension size (the "hidden dimension").
            FFN_size (int): Hidden dimension size between the two linear layers.
        """
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        self.W_1 = nn.Linear(in_features=seq_len * d_model,
                             out_features=FFN_size,
                             bias=False)
        self.W_2 = nn.Linear(in_features=FFN_size,
                             out_features=seq_len * d_model,
                             bias=False)
        self.gelu = nn.GELU()

    def forward(self, X: torch.Tensor):
        # Equation in paragraph below Equation 9 from the paper
        return self.W_2(self.gelu(self.W_1(nn.Flatten(X))))\
                    .reshape(-1, self.seq_len, self.d_model)


class RetNetBlock(nn.Module):
    def __init__(self,
                 seq_len: int,
                 d_model: int,
                 FFN_size: int,
                 head_dim: int):
        """
        Args:
            seq_len (int): Size of context window.
            d_model (int): Embedding dimension size (the "hidden dimension" or
                "hidden size").
            FFN_size (int): Dimension size of the hidden layer in the FFN.
            head_dim (int): Dimension of retention heads. Must be a divisor of
                            d_model.
        """
        super().__init__()

        self.MSR = MSR(h=d_model / head_dim, d_model=d_model)
        self.FFN = FFN(seq_len=seq_len, d_model=d_model, FFN_size=FFN_size)
        self.LN = nn.LayerNorm(d_model)

    def forward(self, Xsupl: torch.Tensor):
        # Equation 9 from the paper
        Ysupl = self.MSR(self.LN(Xsupl)) + Xsupl
        return self.FFN(self.LN(Ysupl)) + Ysupl


class RetNet(nn.Module):
    def __init__(self,
                 seq_len: int,
                 d_model: int,
                 L: int,
                 FFN_size: int,
                 head_dim: int):
        """
        Args:
            seq_len (int): Size of context window.
            d_model (int): Embedding dimension size (the "hidden dimension" or
                "hidden size").
            L (int): Number of stacked RetNetBlocks in the model (the "layers").
            FFN_size (int): Dimension size of the hidden layer in the FFN.
            head_dim (int): Dimension of retention heads. Must be a divisor of
                d_model.
        """
        super().__init__()

        self.model = nn.Sequential(
                *(RetNetBlock(seq_len, d_model, FFN_size, head_dim) for i in range(L)))

    def forward(self, Xsup0: torch.Tensor):
        # Compute contextualized vector representations, from end of first
        # paragraph of Section 2 of the paper
        return self.model(Xsup0)
