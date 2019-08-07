import torch.nn as nn

"""
Their data points were much larger - targeting police beats (which can be relevant seeing that police is spread out 
    proportionately to the population density)
They attempt multi-class (>2) classification to determine the level of crimes - 8 bins with max value of 20
Check out the notebook to see how the training is set upt with their detached states - used to perform TBPTT
"""


class StecKlabjanGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(StecKlabjanGRU, self).__init__()

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=False)  # note not batch first

        self.linear_out = nn.Linear(hidden_size, output_size)

    #         self.init_weights()

    def init_weights(self):
        # should init gru all to zero
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):  # give seq_jump to know which hidden state to use for next sequence
        #         h = torch.random
        # Forward propagate RNN
        out = self.linear_in(x)

        out, h = self.gru(out)  # Not saving hidden state because seq_jump # init lstm weight to 1 to make easier

        out = self.linear_out(out)
        return out, h  # if we never send h its never detached
