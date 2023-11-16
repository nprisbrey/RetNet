import torch
from torch import nn

class RetNetBlock(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        """
        Args:
            seq_len (int): Size of context window.
            d_model (int): Embedding dimension size (the "hidden dimension").
        """
        super().__init__()

        self.MSR = MSR()
        self.FFN = FFN()
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
