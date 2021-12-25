import logging as log
import numpy as np
import torch
import torch.nn as nn

from dataloaders.grid_loader import GridBatchLoader
from utils.configs import BaseConf

"""
General notes on module:
========================
### STRes-Net (add link to paper)
* **Input Data Format:** (N,C,H,W) where C a.k.a the channels is the previous time steps leading up to t
* **Input Data Type:** Continuous value (number of crimes per cell)
* **Output Data Format:** (N,C,H,W) 
* **Output Data Type:** Continuous value (number of crimes per cell)
* **Loss Function:** RMSE
"""


class ResUnit(nn.Module):
    def __init__(self, n_channels=1):
        super(ResUnit, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3,), stride=(1,),
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=(3,), stride=(1,),
                               padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.bn2 = nn.BatchNorm2d(n_channels)

    # using pytorch default weight inits (Xavier Init) we should probably use nn.init.kaiming_normal_
    #         self.init_weights()

    def init_weights(self):
        # torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv1.weight.data.uniform_(-0.5, 0.5)
        self.conv2.bias.data.fill_(0)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
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
        """
        n_layers: number of ResUnits
        in_channels: number of channels at conv1 input 
        n_channels: number of channels at conv1 output and res-units inputs
        conv2 take n_channels and outputs 1 channel
        """
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=n_channels, kernel_size=(3,), stride=(1,),
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=(3,), stride=(1,), padding=1)
        self.resUnits = nn.Sequential()
        for i in range(n_layers):
            self.resUnits.add_module(name='ResUnit' + str(i), module=ResUnit(n_channels))

        # self.init_weights()

    def init_weights(self):
        # torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv1.weight.data.uniform_(-0.5, 0.5)
        self.conv2.bias.data.fill_(0)
        # torch.nn.init.xavier_uniform(self.conv2.weight)
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
        # torch.nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.weight.data.uniform_(-0.5, 0.5)
        self.fc1.bias.input_data.fill_(0)
        # torch.nn.init.xavier_uniform(self.fc2.weight)
        self.fc2.weight.data.uniform_(-0.5, 0.5)
        self.fc2.bias.input_data.fill_(0)

    def forward(self, a):
        Xext = self.fc2(self.relu(self.fc1(a)))
        return Xext.view(-1, 1, self.y_size, self.x_size)


class Fuse(nn.Module):  # fuse the 3 matrices with parametric matrices
    def __init__(self, y_size, x_size):
        super(Fuse, self).__init__()

        self.Wc = nn.parameter.Parameter(torch.zeros(y_size, x_size), requires_grad=True)
        self.Wp = nn.parameter.Parameter(torch.zeros(y_size, x_size), requires_grad=True)
        self.Wq = nn.parameter.Parameter(torch.zeros(y_size, x_size), requires_grad=True)

        # self.init_weights()

    def init_weights(self):
        self.Wc.data.uniform_(-0.5, 0.5)
        self.Wp.data.uniform_(-0.5, 0.5)
        self.Wq.data.uniform_(-0.5, 0.5)

    def forward(self, Xc, Xp, Xq):
        Xres = self.Wc * Xc + self.Wp * Xp + self.Wq * Xq
        return Xres


class STResNetOLD(nn.Module):
    def __init__(self, n_layers, y_size, x_size, lc=1, lp=1, lq=1, n_channels=1, n_ext_features=10):
        """
        n_layers: number of layers
        y_size: grids.shape[-2]
        x_size: grids.shape[-1]
        ext_features: number of external features, dimensions of E
        """

        # TODO: check if pytorch has parallel modules like sequential
        # TODO: See if we can set parallel networks by a param: not just lc,lp,lq, but even more
        # TODO: Add option with no external data

        super(STResNetOLD, self).__init__()
        self.resNetc = ResNet(n_layers, in_channels=lc, n_channels=n_channels)
        self.resNetp = ResNet(n_layers, in_channels=lp, n_channels=n_channels)
        self.resNetq = ResNet(n_layers, in_channels=lq, n_channels=n_channels)
        self.extNet = ExternalNet(in_features=n_ext_features, y_size=y_size, x_size=x_size)
        self.fuse = Fuse(y_size=y_size, x_size=x_size)

    def forward(self, Sc, Sp, Sq, Et=None):
        """
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


class STResNet(nn.Module):
    # todo add docs
    # todo use model_arch instead

    def __init__(self,
                 n_layers,
                 y_size,
                 x_size,
                 n_channels=1,
                 lc=1,
                 lp=1,
                 lq=1,
                 n_ext_features=10,
                 ):
        """
        n_layers: number of layers
        y_size: grids.shape[-2]
        x_size: grids.shape[-1]
        ext_features: number of external features, dimensions of E
        """

        super(STResNet, self).__init__()
        self.name = "ST-RESNET"

        self.res_net_c = ResNet(n_layers, in_channels=lc, n_channels=n_channels)
        self.res_net_p = ResNet(n_layers, in_channels=lp, n_channels=n_channels)
        self.res_net_q = ResNet(n_layers, in_channels=lq, n_channels=n_channels)
        self.ext_net = ExternalNet(in_features=n_ext_features, y_size=y_size, x_size=x_size)
        self.fuse = Fuse(y_size=y_size, x_size=x_size)

    def forward(self, seq_c, seq_p, seq_q, seq_e=None):
        # todo: redo comments for function
        """
        Inputs:
        =======
        Sc, Sp, Sq: Sequence of grids - each grid as a channel
        Et: External features at time t

        Outputs:
        ========
        Xt_hat: Estimated crime grid at time t
        """
        # l indicates the output of the lth ResUnit
        x_c = self.res_net_c(seq_c)
        x_p = self.res_net_p(seq_p)
        x_q = self.res_net_q(seq_q)
        x_res = self.fuse(x_c, x_p, x_q)

        # if seq_e is not None:  # time and weather vectors - weather is not used a.t.m. because of missing data
        #     x_ext = self.ext_net(seq_e)
        #     x_res = x_res * x_ext

        # tanh squeezes values between -1 and 1 that's, that's why the cum-sum wasn't working
        # Last layer is tanh
        if seq_e is None:
            x_t_hat = torch.sigmoid(x_res)
        else:
            x_ext = self.ext_net(seq_e)
            # todo: ask why are we adding - why not multiply
            # x_t_hat = torch.tanh(x_res + x_ext)  # output values can only be between -1,1 for tanh
            x_t_hat = torch.sigmoid(x_res + x_ext)  # target values are between 0,1

        return x_t_hat


class STResNetExtra(nn.Module):
    """
    Extra conv nets to include the google street view, and the demographics
    """

    def __init__(self,
                 n_layers,
                 y_size,
                 x_size,
                 n_channels=1,
                 lc=1,
                 lp=1,
                 lq=1,

                 n_ext_features=10,

                 n_demog_features=37,
                 n_demog_channels=10,
                 n_demog_layers=3,

                 n_gsv_features=512,  # gsv = GOOGLE STREET VIEW
                 n_gsv_channels=10,
                 n_gsv_layers=3,

                 ):
        """
        n_layers: number of layers
        y_size: grids.shape[-2]
        x_size: grids.shape[-1]
        ext_features: number of external features, dimensions of E
        """

        super(STResNetExtra, self).__init__()
        self.name = "ST-RESNET-Extra"

        self.res_net_c = ResNet(n_layers, in_channels=lc, n_channels=n_channels)
        self.res_net_p = ResNet(n_layers, in_channels=lp, n_channels=n_channels)
        self.res_net_q = ResNet(n_layers, in_channels=lq, n_channels=n_channels)
        self.res_net_demog = ResNet(n_demog_layers, in_channels=n_demog_features, n_channels=n_demog_channels)
        self.res_net_street_view = ResNet(n_gsv_layers, in_channels=n_gsv_features, n_channels=n_gsv_channels)
        self.ext_net = ExternalNet(in_features=n_ext_features, y_size=y_size, x_size=x_size)
        self.fuse = Fuse(y_size=y_size, x_size=x_size)

    def forward(self, seq_c, seq_p, seq_q, seq_e=None, seq_demog=None, seq_gsv=None):
        # todo: redo comments for function
        """
        Inputs:
        =======
        Sc, Sp, Sq: Sequence of grids - each grid as a channel
        Et: External features at time t

        Outputs:
        ========
        Xt_hat: Estimated crime grid at time t
        """
        # l indicates the output of the lth ResUnit

        x_c = self.res_net_c(seq_c)
        x_p = self.res_net_p(seq_p)
        x_q = self.res_net_q(seq_q)
        x_res = self.fuse(x_c, x_p, x_q)

        if seq_demog is not None:
            x_demog = self.res_net_demog(seq_demog)
            x_res = x_res * x_demog

        if seq_gsv is not None:
            x_gsv = self.res_net_street_view(seq_gsv)
            x_res = x_res * x_gsv
        #
        # if seq_e:  # time and weather vectors - weather is not used a.t.m. because of missing data
        #     x_ext = self.ext_net(seq_e)
        #     x_res = x_res * x_ext
        #

        # tanh squeezes values between -1 and 1 that's, that's why the cum-sum wasn't working
        # Last layer is tanh
        if seq_e is None:
            x_t_hat = torch.tanh(x_res)
        else:
            x_ext = self.ext_net(seq_e)
            # todo: ask why are we adding - why not multiply
            # x_t_hat = torch.tanh(x_res + x_ext)  # output values can only be between -1,1 for tanh

            # target values are between 0 and 1
            x_t_hat = torch.sigmoid(x_res + x_ext)  # output values can only be between 0,1 for tanh

        return x_t_hat


# training loops
def train_epoch_for_st_res_net_extra(model, optimiser, batch_loader, loss_fn, total_losses, conf):
    """
    Training the STResNetExtra model for a single epoch
    Grid based loaders will always be regression models
    """
    epoch_losses = []
    num_batches = batch_loader.num_batches

    demog_grid = torch.Tensor(batch_loader.dataset.demog_grid).to(conf.device)
    street_grid = torch.Tensor(batch_loader.dataset.street_grid).to(conf.device)

    for batch_indices, batch_seq_c, batch_seq_p, batch_seq_q, batch_seq_e, batch_seq_t in batch_loader:
        current_batch = batch_loader.current_batch

        batch_seq_c = torch.Tensor(batch_seq_c).to(conf.device)
        batch_seq_p = torch.Tensor(batch_seq_p).to(conf.device)
        batch_seq_q = torch.Tensor(batch_seq_q).to(conf.device)
        batch_seq_e = torch.Tensor(batch_seq_e).to(conf.device)
        batch_seq_t = torch.Tensor(batch_seq_t).to(conf.device)

        batch_pred = model(seq_c=batch_seq_c,
                           seq_p=batch_seq_p,
                           seq_q=batch_seq_q,
                           seq_e=batch_seq_e,
                           seq_demog=demog_grid,
                           seq_gsv=street_grid)

        # might need to flatten
        loss = loss_fn(input=batch_pred, target=batch_seq_t)
        epoch_losses.append(loss.item())
        total_losses.append(epoch_losses[-1])

        if model.training:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            log.debug(f"Batch: {current_batch:04d}/{num_batches:04d} \t Loss: {epoch_losses[-1]:.4f}")
    mean_epoch_loss = np.mean(epoch_losses)

    return mean_epoch_loss


def train_epoch_for_st_res_net(model, optimiser, batch_loader, loss_fn, total_losses, conf):
    """
    Training the STResNet model for a single epoch
    """
    epoch_losses = []
    num_batches = batch_loader.num_batches

    for batch_indices, batch_seq_c, batch_seq_p, batch_seq_q, batch_seq_e, batch_seq_t in batch_loader:
        current_batch = batch_loader.current_batch

        batch_seq_c = torch.Tensor(batch_seq_c).to(conf.device)
        batch_seq_p = torch.Tensor(batch_seq_p).to(conf.device)
        batch_seq_q = torch.Tensor(batch_seq_q).to(conf.device)
        batch_seq_e = torch.Tensor(batch_seq_e).to(conf.device)
        batch_seq_t = torch.Tensor(batch_seq_t).to(conf.device)

        batch_pred = model(seq_c=batch_seq_c,
                           seq_p=batch_seq_p,
                           seq_q=batch_seq_q,
                           seq_e=batch_seq_e)

        # might need to flatten
        loss = loss_fn(input=batch_pred, target=batch_seq_t)
        epoch_losses.append(loss.item())
        total_losses.append(epoch_losses[-1])

        if model.training:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            log.debug(f"Batch: {current_batch:04d}/{num_batches:04d} \t Loss: {epoch_losses[-1]:.4f}")

    mean_epoch_loss = np.mean(epoch_losses)

    return mean_epoch_loss


# evaluation loops
def evaluate_st_res_net_extra(model: nn.Module, batch_loader, conf):
    """
    st_res_net model looks at historic crime data, time vector data as well as demographic and street view data

    :param model: model that takes (seq_c, seq_p, seq_q, seq_e, seq_demog, seq_gsv)
    :param batch_loader: iterates with outputs: (batch_indices, batch_seq_c, batch_seq_p, batch_seq_q, batch_seq_e, batch_seq_t)
    :param conf: only used the device attached to conf to use cuda if available
    :return: y_count, y_class, y_score, t_range
        - y_count: normalized target crime counts for regression models
        - y_class: class labels {0, 1}
        - y_score: float value representing the probability of crime happening

    Notes
    -----
    Only used to get probas in a time and location based format. The hard predictions should be done outside
    this function where the threshold is determined using only the training data
    """
    y_score = np.zeros(batch_loader.dataset.target_shape, dtype=np.float)
    y_count = batch_loader.dataset.targets[-len(y_score):]
    y_class = batch_loader.dataset.labels[-len(y_score):]
    # y_class = np.copy(y_count) # batch_loader.dataset.labels[-len(y_score):]
    # if y_class.min() < 0:
    #     raise ValueError(f"Data must be normalised between (0,1), min value is {y_class.min()}")
    # y_class[y_class > 0] = 1  # ensure normalisation between 0 and 1 not -1 and 1

    t_range = batch_loader.dataset.t_range[-len(y_score):]

    with torch.set_grad_enabled(False):
        model.eval()

        num_batches = batch_loader.num_batches

        demog_grid = torch.Tensor(batch_loader.dataset.demog_grid).to(conf.device)
        street_grid = torch.Tensor(batch_loader.dataset.street_grid).to(conf.device)

        for batch_indices, batch_seq_c, batch_seq_p, batch_seq_q, batch_seq_e, batch_seq_t in batch_loader:
            current_batch = batch_loader.current_batch

            batch_seq_c = torch.Tensor(batch_seq_c).to(conf.device)
            batch_seq_p = torch.Tensor(batch_seq_p).to(conf.device)
            batch_seq_q = torch.Tensor(batch_seq_q).to(conf.device)
            batch_seq_e = torch.Tensor(batch_seq_e).to(conf.device)
            batch_seq_t = torch.Tensor(batch_seq_t).to(conf.device)

            batch_probas_pred = model(seq_c=batch_seq_c,
                                      seq_p=batch_seq_p,
                                      seq_q=batch_seq_q,
                                      seq_e=batch_seq_e,
                                      seq_demog=demog_grid,
                                      seq_gsv=street_grid)

            for i, p in zip(batch_indices, batch_probas_pred.cpu().numpy()):
                y_score[i] = p

    return y_count, y_class, y_score, t_range


def evaluate_st_res_net(model, batch_loader: GridBatchLoader, conf: BaseConf):
    """
    st_res_net model only looks at historic crime data as well as time vector data

    :param model: model that takes (seq_c, seq_p, seq_q, seq_e)
    :param batch_loader: iterates with outputs: (batch_indices, batch_seq_c, batch_seq_p, batch_seq_q, batch_seq_e, batch_seq_t)
    :param conf: only used the device attached to conf to use cuda if available
    :return: y_count, y_class, y_score, t_range




    Notes
    -----
    Only used to get probas in a time and location based format. The hard predictions should be done outside
    this function where the threshold is determined using only the training data
    """

    y_score = np.zeros(batch_loader.dataset.target_shape, dtype=np.float)
    y_count = batch_loader.dataset.targets[-len(y_score):]
    y_class = batch_loader.dataset.labels[-len(y_score):]
    # y_class = np.copy(y_count)
    # if y_class.min() < 0:
    #     raise ValueError(f"Data must be normalised between (0,1), min value is {y_class.min()}")
    # y_class[y_class > 0] = 1

    t_range = batch_loader.dataset.t_range[-len(y_score):]

    with torch.set_grad_enabled(False):
        model.eval()

        num_batches = batch_loader.num_batches

        demog_grid = torch.Tensor(batch_loader.dataset.demog_grid).to(conf.device)
        street_grid = torch.Tensor(batch_loader.dataset.street_grid).to(conf.device)

        for batch_indices, batch_seq_c, batch_seq_p, batch_seq_q, batch_seq_e, batch_seq_t in batch_loader:
            current_batch = batch_loader.current_batch

            batch_seq_c = torch.Tensor(batch_seq_c).to(conf.device)
            batch_seq_p = torch.Tensor(batch_seq_p).to(conf.device)
            batch_seq_q = torch.Tensor(batch_seq_q).to(conf.device)
            batch_seq_e = torch.Tensor(batch_seq_e).to(conf.device)
            batch_seq_t = torch.Tensor(batch_seq_t).to(conf.device)

            batch_probas_pred = model(seq_c=batch_seq_c,
                                      seq_p=batch_seq_p,
                                      seq_q=batch_seq_q,
                                      seq_e=batch_seq_e)

            for i, p in zip(batch_indices, batch_probas_pred.cpu().numpy()):
                y_score[i] = p

    return y_count, y_class, y_score, t_range
