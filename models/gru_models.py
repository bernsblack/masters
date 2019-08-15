from torch import nn

"""
### GRU (Multi-input single output)
Like a grid of 5 by 5 as input trying to predict the center cell for the next time step
* **Input Data Format:** (N,C,H,W) where C a.k.a the channels is the previous time steps leading up to t
* **Input Data Type:** Continuous value (number of crimes per cell)
* **Output Data Format:** (N,C,H,W)
* **Output Data Type:** Continuous value (number of crimes per cell)
* **Loss Function:** RMSE
"""


class GRUMLPRegressor(nn.Module):
    """
    GRU then an MLP
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUMLPRegressor, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # note batch first
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, h0=None):
        # Forward propagate RNN
        if h0 is not None:
            out, hn = self.gru(x, h0)
        else:
            out, hn = self.gru(x)  # hidden state start is zero
        out = self.lin1(out)
        out = self.relu(out)
        out = self.lin2(out)
        # softmax?

        # Reshape output to (batch_size*seq_len, hidden_size)
        #         out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))

        # Decode hidden states of all time step

        return out  # if we never send h its never detached


class GRUMLPClassifier(nn.Module):
    """
    GRU then a MLP (flattens data output to easier use in NLLLoss)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUMLPClassifier, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # note batch first
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, h0=None):
        if h0 is not None:
            out, hn = self.gru(x, h0)
        else:
            out, hn = self.gru(x)  # hidden state start is zero
        out = self.lin1(out)
        out = self.relu(out)
        out = self.lin2(out)

        # Reshape output to (batch_size*seq_len, hidden_size)
        out = out.contiguous().view(out.size(0) * out.size(1), out.size(2))

        return out  # if we never send h its never detached
