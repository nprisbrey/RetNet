import torch
from torch import nn

class FNN(nn.Module):
    """ Feed Forward Network """
    def __init__(self, seq_len: int, d_model: int, fnn_hidden_dim: int=None):
        """
        Args:
            seq_len (int): Size of context window.
            d_model (int): Embedding dimension size (the "hidden dimension").
            fnn_hidden_dim (int): Hidden dimension size between the two linear
                layers. If not set, will use seq_len * d_model * 2.
        """
        super().__init__()

        if fnn_hidden_dim is None:
            fnn_hidden_dim = seq_len * d_model * 2

        self.W_1 = nn.Linear(in_features=seq_len * d_model,
                             out_features=fnn_hidden_dim,
                             bias=False)
        self.W_2 = nn.Linear(in_features=fnn_hidden_dim,
                             out_features=seq_len * d_model,
                             bias=False)
        self.gelu = nn.GELU()

    def forward(self, X: torch.Tensor):
        # Equation in paragraph below Equation 9 from the paper
        return self.W_2(self.gelu(self.W_1(nn.Flatten(X)))).reshape(-1,
                                                                    seq_len,
                                                                    d_model)


class RetNetBlock(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        """
        Args:
            seq_len (int): Size of context window.
            d_model (int): Embedding dimension size (the "hidden dimension").
        """
        super().__init__()

        self.MSR = MSR()
        self.FFN = FFN(seq_len=seq_len, d_model=d_model)
        self.LN = nn.LayerNorm(d_model)

    def forward(self, Xsupl: torch.Tensor):
        # Equation 9 from the paper
        Ysupl = self.MSR(self.LN(Xsupl)) + Xsupl
        return self.FFN(self.LN(Ysupl)) + Ysupl


class RetNet(nn.Module):
    def __init__(self, seq_len: int, d_model: int, L: int=1):
        """
        Args:
            seq_len (int): Size of context window.
            d_model (int): Embedding dimension size (the "hidden dimension").
            L (int): Number of stacked RetNetBlocks in the model.
        """
        super().__init__()

        self.model = nn.Sequential(
                [RetNetBlock(seq_len, d_model) for i in range(L)])

    def forward(self, Xsup0: torch.Tensor):
        # Compute contextualized vector representations, from end of first
        # paragraph of Section 2 of the paper
        return self.model(Xsup0)
