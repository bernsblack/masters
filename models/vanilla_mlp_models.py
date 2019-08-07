from torch import nn

"""
### Vanilla MLP Classifier
* **Input Data Format:** (N,D) N: number of data D: number of dimensions
* **Input Data Type:** Continuous value
* **Output Data Format:** (N,C) N: number of data C: number of classes 
* **Output Data Type:** Continuous value (number of crimes per cell)
* **Loss Function:** NLLLikelihood (need to logSoftmax first) **OR** Cross Entropy Loss (includes logSoftmax) 
"""


class MLPClassifier(nn.Module):
    """
    MLP Classifier
    """

    def __init__(self, input_size=784, hidden_size=100, n_classes=10, num_layers=1):
        super(MLPClassifier, self).__init__()

        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        out = self.logSoftmax(out)

        return out
