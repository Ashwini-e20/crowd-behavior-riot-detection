import torch
import torch.nn as nn

class GRUModel(nn.Module):
    """
    Defines the GRU model architecture.
    """
    def __init__(self, input_size=1280, hidden_size=256, num_layers=2, output_size=2, dropout=0.3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        # We only need the output of the last time step
        return self.linear(gru_out[:, -1, :])

