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
