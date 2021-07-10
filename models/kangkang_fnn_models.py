import logging as log
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils.configs import BaseConf

"""
### Feed Forward Network (Kang and Kang)

* **Input Data Format:** (N,D_spatial),(N,D_temporal),(N,D_environment) or (N,D) vetors are fed in independently 
* **Input Data Type:** Continuous values
Spatial feature group (35-D): demographic (9-D), housing (6-D), education (8-D), and economic (12-D).
 - we included the area of the tract - to hopefull get a density factor in as well.
  
Temporal feature group (15-D): weather (11-D), number of incidents of crime occurrence by sampling point in 2013 (1-D),
 number of incidents of crime occurrence by census tract in 2013 (1-D), number of incidents of crime occurrence by date
  in 2013 (1-D), and number of incidents of crime occurrence by census tract yesterday (1-D).

Environmental context feature group (4096-D): an image feature (4096-D).

Key extraction of the data will be done in the dataset / data loader.

* **Output Data Format:** (N,C) with C being a binary class 
* **Output Data Type:** Continuous value (number of crimes per cell)
* **Loss Function:** RMSE
"""


class KangFeedForwardNetwork(nn.Module):
    def __init__(self, spc_size=37, tmp_size=15, env_size=512, output_size=1, dropout_p=0.5, model_arch=None):
        """

        :param spc_size: spatial dimension vector size
        :param tmp_size: temporal dimension vector size
        :param env_size: environmental dimension vector size
        :param output_size: 1 for regression (default) and 2 for classification
        :param dropout_p: dropout probability
        :param model_arch: model architecture dictionary
        """

        super(KangFeedForwardNetwork, self).__init__()

        self.name = "KangFNN"
        # drop out is not saved on the model state_dict - remember to turn off in evaluation
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        if model_arch:
            h_size0 = model_arch.get("h_size0")
            h_size1 = model_arch.get("h_size1")
            h_size2 = model_arch.get("h_size2")

            scp_net_h0 = model_arch.get("scp_net_h0")
            scp_net_h1 = model_arch.get("scp_net_h1")
            tmp_net_h0 = model_arch.get("tmp_net_h0")
            tmp_net_h1 = model_arch.get("tmp_net_h1")
            env_net_h0 = model_arch.get("env_net_h0")
            env_net_h1 = model_arch.get("env_net_h1")

            final_net_h0 = scp_net_h1 + tmp_net_h1 + env_net_h1
            final_net_h1 = model_arch.get("final_net_h1")
        else:  # default architecture
            scp_net_h0 = 256
            scp_net_h1 = 128
            tmp_net_h0 = 256
            tmp_net_h1 = 128
            env_net_h0 = 256
            env_net_h1 = 128

            final_net_h0 = scp_net_h1 + tmp_net_h1 + env_net_h1
            final_net_h1 = 1024

        self.spcNet = nn.Sequential(nn.Linear(spc_size, scp_net_h0),
                                    nn.ReLU(),
                                    nn.Linear(scp_net_h0, scp_net_h0),
                                    nn.ReLU(),
                                    nn.Linear(scp_net_h0, scp_net_h1),
                                    nn.ReLU())

        self.tmpNet = nn.Sequential(nn.Linear(tmp_size, tmp_net_h0),
                                    nn.ReLU(),
                                    nn.Linear(tmp_net_h0, tmp_net_h0),
                                    nn.ReLU(),
                                    nn.Linear(tmp_net_h0, tmp_net_h1),
                                    nn.ReLU())

        self.envNet = nn.Sequential(nn.Linear(env_size, env_net_h0),
                                    nn.ReLU(),
                                    nn.Linear(env_net_h0, env_net_h0),
                                    nn.ReLU(),
                                    nn.Linear(env_net_h0, env_net_h1),
                                    nn.ReLU())

        self.finalNet = nn.Sequential(nn.Linear(final_net_h0, final_net_h1),
                                      nn.ReLU(),
                                      nn.Linear(final_net_h1, final_net_h1),
                                      nn.ReLU(),
                                      nn.Linear(final_net_h1, output_size))

    def forward(self, spc_vec, tmp_vec, env_vec):
        if self.dropout_p > 0:
            spc_vec = self.dropout(spc_vec)
            tmp_vec = self.dropout(tmp_vec)
            env_vec = self.dropout(env_vec)

        mid_vec = torch.cat([self.spcNet(spc_vec), self.tmpNet(tmp_vec), self.envNet(env_vec)], dim=-1)
        out_vec = self.finalNet(mid_vec)

        return out_vec


class SimpleKangFNN(nn.Module):
    def __init__(self, spc_size=37, tmp_size=15, env_size=512, output_size=1, dropout_p=0.5, model_arch=None):
        """

        :param spc_size: spatial dimension vector size
        :param tmp_size: temporal dimension vector size
        :param env_size: environmental dimension vector size
        :param output_size: 1 for regression (default) and 2 for classification
        :param dropout_p: dropout probability
        :param model_arch: model architecture dictionary
        """

        super(SimpleKangFNN, self).__init__()

        self.name = "SimpleKangFNN"

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)  # not used when model is in eval mode

        if model_arch:
            h_size0 = model_arch.get("h_size0")
            h_size1 = model_arch.get("h_size1")
            h_size2 = model_arch.get("h_size2")
        else:
            h_size0 = 50
            h_size1 = 50
            h_size2 = 50

        self.spcNet = nn.Sequential(nn.Linear(spc_size, h_size0),
                                    nn.ReLU(),
                                    nn.Linear(h_size0, h_size1),
                                    nn.ReLU())

        self.tmpNet = nn.Sequential(nn.Linear(tmp_size, h_size0),
                                    nn.ReLU(),
                                    nn.Linear(h_size0, h_size1),
                                    nn.ReLU())

        self.envNet = nn.Sequential(nn.Linear(env_size, h_size0),
                                    nn.ReLU(),
                                    nn.Linear(h_size0, h_size1),
                                    nn.ReLU())

        self.finalNet = nn.Sequential(nn.Linear(3 * h_size1, h_size2),
                                      nn.ReLU(),
                                      nn.Linear(h_size2, output_size))

    def forward(self, spc_vec, tmp_vec, env_vec):
        if self.dropout_p > 0:
            spc_vec = self.dropout(spc_vec)
            tmp_vec = self.dropout(tmp_vec)
            env_vec = self.dropout(env_vec)

        mid_vec = torch.cat([self.spcNet(spc_vec), self.tmpNet(tmp_vec), self.envNet(env_vec)], dim=-1)
        out_vec = self.finalNet(mid_vec)

        return out_vec


# training loops
def train_epoch_for_fnn(model, optimiser, batch_loader, loss_fn, total_losses, conf: BaseConf):
    """
    Training the FNN model for a single epoch
    """
    epoch_losses = []
    num_batches = batch_loader.num_batches
    for indices, spc_feats, tmp_feats, env_feats, targets, labels in batch_loader:
        current_batch = batch_loader.current_batch

        # Transfer to PyTorch Tensor and GPU
        spc_feats = torch.Tensor(spc_feats[-1]).to(conf.device)  # only taking [-1] for fnn
        tmp_feats = torch.Tensor(tmp_feats[-1]).to(conf.device)  # only taking [-1] for fnn
        env_feats = torch.Tensor(env_feats[-1]).to(conf.device)  # only taking [-1] for fnn

        out = model(spc_feats, tmp_feats, env_feats)

        if conf.use_classification:
            labels = torch.LongTensor(labels[-1, :, 0]).to(conf.device)  # only taking [-1] for fnn
            loss = loss_fn(input=out, target=labels)
        else:
            targets = torch.FloatTensor(targets[-1, :]).to(conf.device)  # only taking [-1] for fnn
            loss = loss_fn(input=out, target=targets)

        epoch_losses.append(loss.item())
        total_losses.append(epoch_losses[-1])

        if model.training:  # not used in validation loops
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            log.debug(f"Batch: {current_batch:04d}/{num_batches:04d} \t Loss: {epoch_losses[-1]:.4f}")
    mean_epoch_loss = np.mean(epoch_losses)
    return mean_epoch_loss


# evaluation loops
def evaluate_fnn(model, batch_loader, conf):
    """
    :param model: any torch model that takes spc_feats, tmp_feats, env_feats as features
    :param batch_loader: batch loader that is iterable -> returns: indices, spc_feats, tmp_feats, env_feats
    :param conf: only value used it the device attached to the conf variable
    :return: y_count, y_class, y_score, t_range
        - y_count: scaled target counts
        - y_class: class label (crime/not-crime)
        - y_score: model output (probability of crime [classification] or scaled crime count [regression])
        - t_range: time range

    Notes
    -----
    Only used to get probas in a time and location based format. The hard predictions should be done outside
    this function where the threshold is determined using only the training data
    """
    y_score = np.zeros(batch_loader.dataset.target_shape, dtype=np.float)  # probas_pred
    y_count = batch_loader.dataset.targets[-len(y_score):]
    y_class = batch_loader.dataset.labels[-len(y_score):]
    t_range = batch_loader.dataset.t_range[-len(y_score):]

    with torch.set_grad_enabled(False):
        model.eval()

        num_batches = batch_loader.num_batches
        for indices, spc_feats, tmp_feats, env_feats, targets, labels in batch_loader:
            current_batch = batch_loader.current_batch

            # Transfer to PyTorch Tensor and GPU
            spc_feats = torch.Tensor(spc_feats[-1]).to(conf.device)  # only taking [-1] for fnn
            tmp_feats = torch.Tensor(tmp_feats[-1]).to(conf.device)  # only taking [-1] for fnn
            env_feats = torch.Tensor(env_feats[-1]).to(conf.device)  # only taking [-1] for fnn
            # targets = torch.LongTensor(targets[0, :, 0]).to(conf.device)  # only taking [0] for fnn
            out = model(spc_feats, tmp_feats, env_feats)

            if conf.use_classification:
                batch_y_score = F.softmax(out, dim=-1)[:, 1].cpu().numpy()  # select class1 prediction
            else:
                batch_y_score = out.cpu().numpy()  # select class1 prediction

            for i, p in zip(indices, batch_y_score):
                n, c, l = i
                y_score[n, c, l] = p

    return y_count, y_class, y_score, t_range
