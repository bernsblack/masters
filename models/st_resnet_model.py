import torch
import torch.nn as nn

"""
General notes on module:
========================
### STRes-Net (add link to paper)
* **Input Data Format:** (N,C,W,H) where C a.k.a the channels is the previous time steps leading up to t
* **Input Data Type:** Continuous value (number of crimes per cell)
* **Output Data Format:** (N,C,W,H) 
* **Output Data Type:** Continuous value (number of crimes per cell)
* **Loss Function:** RMSE
"""


class ResUnit(nn.Module):
    def __init__(self, n_channels=1):
        super(ResUnit, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.bn2 = nn.BatchNorm2d(n_channels)

    # using pytorch default weight inits (Xavier Init) we should probably use nn.init.kaiming_normal_
    #         self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.uniform_(-0.5, 0.5)
        self.conv2.bias.data.fill_(0)
        self.conv1.weight.data.uniform_(-0.5, 0.5)
        self.conv2.bias.data.fill_(0)

    def forward(self, x):
        o = self.bn1(x)
        o = self.relu(o)
        o = self.conv1(o)
        o = self.bn2(o)
        o = self.relu(o)
        o = self.conv2(o)
        o = x + o

        return o


class ResNet(nn.Module):
    def __init__(self, n_layers=1, in_channels=1, n_channels=1):
        r"""
        n_layers: number of ResUnits
        in_channels: number of channels at conv1 input 
        n_channels: number of channels at conv1 output and res-units inputs
        conv2 take n_channels and outputs 1 channel
        """
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.resUnits = nn.Sequential()
        for i in range(n_layers):
            self.resUnits.add_module(name='ResUnit' + str(i), module=ResUnit(n_channels))

        # self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.uniform_(-0.5, 0.5)
        self.conv2.bias.data.fill_(0)
        self.conv1.weight.data.uniform_(-0.5, 0.5)
        self.conv2.bias.data.fill_(0)

    def forward(self, x):
        o = self.conv1(x)
        o = self.resUnits(o)
        o = self.conv2(o)

        return o


class ExternalNet(nn.Module):  # need to add 
    def __init__(self, in_features, y_size, x_size):
        super(ExternalNet, self).__init__()

        self.y_size = y_size
        self.x_size = x_size
        self.out_features = y_size * x_size
        self.fc1 = nn.Linear(in_features, self.out_features)
        self.fc2 = nn.Linear(self.out_features, self.out_features)
        self.relu = nn.ReLU()

        # self.init_weights()

    def init_weights(self):
        self.fc1.weight.data.uniform_(-0.5, 0.5)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.5, 0.5)
        self.fc2.bias.data.fill_(0)

    def forward(self, a):
        Xext = self.fc2(self.relu(self.fc1(a)))
        return Xext.view(-1, 1, self.y_size, self.x_size)


class Fuse(nn.Module):  # fuse the 3 matrices with parametric matrices
    def __init__(self, y_size, x_size):
        super(Fuse, self).__init__()

        self.Wc = nn.parameter.Parameter(torch.zeros(y_size, x_size))
        self.Wp = nn.parameter.Parameter(torch.zeros(y_size, x_size))
        self.Wq = nn.parameter.Parameter(torch.zeros(y_size, x_size))

        # self.init_weights()

    def init_weights(self):
        self.Wc.data.uniform_(-0.5, 0.5)
        self.Wp.data.uniform_(-0.5, 0.5)
        self.Wq.data.uniform_(-0.5, 0.5)

    def forward(self, Xc, Xp, Xq):
        Xres = self.Wc * Xc + self.Wp * Xp + self.Wq * Xq
        return Xres


class STResNet(nn.Module):
    def __init__(self, n_layers, y_size, x_size, lc=1, lp=1, lq=1, n_channels=1, n_ext_features=10):
        r"""
        n_layers: number of layers
        y_size: grids.shape[-2]
        x_size: grids.shape[-1]
        ext_features: number of external features, dimensions of E
        """

        # TODO: check if pytorch has parallel modules like sequential
        # TODO: See if we can set parallel networks by a param: not just lc,lp,lq, but even more
        # TODO: Add option with no external data

        super(STResNet, self).__init__()
        self.resNetc = ResNet(n_layers, in_channels=lc, n_channels=n_channels)
        self.resNetp = ResNet(n_layers, in_channels=lp, n_channels=n_channels)
        self.resNetq = ResNet(n_layers, in_channels=lq, n_channels=n_channels)
        self.extNet = ExternalNet(in_features=n_ext_features, y_size=y_size, x_size=x_size)
        self.fuse = Fuse(y_size=y_size, x_size=x_size)

    def forward(self, Sc, Sp, Sq, Et=None):
        r"""
        Inputs:
        =======
        Sc, Sp, Sq: Sequence of grids - each grid as a channel
        Et: External features at time t

        Outputs:
        ========
        Xt_hat: Estimated crime grid at time t
        """
        # l indicates the output of the lth ResUnit
        Xc = self.resNetc(Sc)
        Xp = self.resNetp(Sp)
        Xq = self.resNetq(Sq)
        Xres = self.fuse(Xc, Xp, Xq)

        # sigmoid squeezes values between 0 and 1 that's, that's why the cum-sum wasn't working
        #         Last layer is sigmoid not tanh
        #         our values are all positive no real reason to use tan like the used in deep-st
        if Et is None:
            Xt_hat = torch.sigmoid(Xres)
        else:
            Xext = self.extNet(Et)
            Xt_hat = torch.sigmoid(Xres + Xext)

        return Xt_hat
