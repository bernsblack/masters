from abc import ABC

import torch
from torch import nn
import logging as log
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

"""
### GRU (Multi-input single output)
Like a grid of 5 by 5 as input trying to predict the center cell for the next time step
* **Input Data Format:** (N,C,H,W) where C a.k.a the channels is the previous time steps leading up to t
* **Input Data Type:** Continuous value (number of crimes per cell)
* **Output Data Format:** (N,C,H,W)
* **Output Data Type:** Continuous value (number of crimes per cell)
* **Loss Function:** RMSE
"""


class MultiLayerGRU(nn.Module):
    def __init__(self, input_size=5, h_size0=15, h_size1=10, o_size=2):
        super(MultiLayerGRU, self).__init__()

        self.name = "MultiLayerGRU"

        self.gru0 = nn.GRU(input_size=input_size,
                           hidden_size=h_size0,
                           num_layers=1)

        self.gru1 = nn.GRU(input_size=h_size0,
                           hidden_size=h_size1,
                           num_layers=1)

        self.gru2 = nn.GRU(input_size=h_size1,
                           hidden_size=o_size,
                           num_layers=1)

    def forward(self, x):
        o0, h0 = self.gru0(x)
        o1, h1 = self.gru1(o0)
        o2, h2 = self.gru2(o1)

        return o2


class GRUFNN1(nn.Module):
    """
    GRU then an FNN
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUFNN1, self).__init__()

        self.name = "GRUFNN"

        self.gru = nn.GRU(input_size, hidden_size, num_layers)  # (seq_len, batch_size, n_features) format
        self.lin1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, h0=None):
        # Forward propagate RNN
        if h0 is not None:
            out, hn = self.gru(x, h0)
        else:
            out, hn = self.gru(x)  # hidden state start is zero
        out = self.relu(out)
        out = self.lin1(out)
        out = self.relu(out)

        # softmax - is applied in the loss function - should then explicitly be used when predicting

        #  only if we wrapper our data from (batch_size*seq_len, hidden_size)
        # Reshape output to (batch_size*seq_len, hidden_size)
        #         out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))

        # Decode hidden states of all time step

        return out  # if we never send h its never detached


class GRUFNN2(nn.Module):
    """
    GRU then an FNN
    """

    def __init__(self, input_size, hidden_size0,hidden_size1, output_size, num_layers=1):
        super(GRUFNN2, self).__init__()

        self.name = "GRUFNN"

        self.gru = nn.GRU(input_size, hidden_size0, num_layers, batch_first=True)  # note batch first
        self.lin1 = nn.Linear(hidden_size0, hidden_size1)
        self.lin2 = nn.Linear(hidden_size1, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, h0=None):
        # Forward propagate RNN
        if h0 is not None:
            out, hn = self.gru(x, h0)
        else:
            out, hn = self.gru(x)  # hidden state start is zero
        out = self.relu(out)
        out = self.lin1(out)
        out = self.relu(out)
        out = self.lin2(out)
        # softmax - is applied in the loss function - should then explicitly be used when predicting

        #  only if we wrapper our data from (batch_size*seq_len, hidden_size)
        # Reshape output to (batch_size*seq_len, hidden_size)
        #         out = out.contiguous().view(out.size(0)*out.size(1), out.size(2))

        # Decode hidden states of all time step

        return out  # if we never send h its never detached


class RecurrentFeedForwardNetwork(nn.Module):
    def __init__(self, spc_size=37, tmp_size=15, env_size=512, dropout_p=0.5, model_arch=None):
        super(RecurrentFeedForwardNetwork, self).__init__()

        # drop out is not saved on the model state_dict - remember to turn off in evaluation
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        if model_arch:
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

        self.tmpNet = GRUFNN2(input_size=tmp_size,
                              hidden_size0=tmp_net_h0,
                              hidden_size1=tmp_net_h0,
                              output_size=tmp_net_h1)

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
                                      nn.Linear(final_net_h1, 2))

    def forward(self, spc_vec, tmp_vec, env_vec):
        if self.dropout_p > 0:
            spc_vec = self.dropout(spc_vec)
            tmp_vec = self.dropout(tmp_vec)
            env_vec = self.dropout(env_vec)

        # only take the very last step of seq_len output from tmpNet
        mid_vec = torch.cat([self.spcNet(spc_vec), self.tmpNet(tmp_vec)[-1], self.envNet(env_vec)], dim=-1)
        out_vec = self.finalNet(mid_vec)

        return out_vec


class SimpleRecurrentFeedForwardNetwork(nn.Module, ABC):
    def __init__(self, spc_size=37, tmp_size=15, env_size=512, dropout_p=0.5, model_arch=None):
        super(SimpleRecurrentFeedForwardNetwork, self).__init__()

        self.name = "Simple RFFN"  # used to setup the model folders

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

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

        self.tmpNet = GRUFNN1(input_size=tmp_size,
                              hidden_size=h_size0,
                              output_size=h_size1)

        self.envNet = nn.Sequential(nn.Linear(env_size, h_size0),
                                    nn.ReLU(),
                                    nn.Linear(h_size0, h_size1),
                                    nn.ReLU())

        self.finalNet = nn.Sequential(nn.Linear(3 * h_size1, h_size2),
                                      nn.ReLU(),
                                      nn.Linear(h_size2, 2))

    def forward(self, spc_vec, tmp_vec, env_vec):
        if self.dropout_p > 0:
            spc_vec = self.dropout(spc_vec)
            tmp_vec = self.dropout(tmp_vec)
            env_vec = self.dropout(env_vec)

        # only take the very last step of seq_len output from tmpNet
        mid_vec = torch.cat([self.spcNet(spc_vec), self.tmpNet(tmp_vec)[-1], self.envNet(env_vec)], dim=-1)
        out_vec = self.finalNet(mid_vec)

        return out_vec


# training loops
def train_epoch_for_rfnn(model, optimiser, batch_loader, loss_fn, total_losses, conf):
    """
    Training the FNN model for a single epoch
    """
    epoch_losses = []
    num_batches = batch_loader.num_batches
    for indices, spc_feats, tmp_feats, env_feats, targets in batch_loader:
        current_batch = batch_loader.current_batch

        # Transfer to PyTorch Tensor and GPU
        spc_feats = torch.Tensor(spc_feats[-1]).to(conf.device)  # only taking [-1] for fnn
        tmp_feats = torch.Tensor(tmp_feats).to(conf.device)  # only taking [-1] for fnn
        env_feats = torch.Tensor(env_feats[-1]).to(conf.device)  # only taking [-1] for fnn
        targets = torch.LongTensor(targets[-1, :, 0]).to(conf.device)  # only taking [-1] for fnn

        out = model(spc_feats, tmp_feats, env_feats)
        loss = loss_fn(input=out, target=targets)
        epoch_losses.append(loss.item())
        total_losses.append(epoch_losses[-1])

        if model.training:  # not used in validation loops
            optimiser.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)  # used as regularisation
            optimiser.step()
            log.debug(f"Batch: {current_batch:04d}/{num_batches:04d} \t Loss: {epoch_losses[-1]:.4f}")
    mean_epoch_loss = np.mean(epoch_losses)
    return mean_epoch_loss


# evaluation loops
def evaluate_rfnn(model, batch_loader, conf):
    """
    Only used to get probas in a time and location based format. The hard predictions should be done outside
    this function where the threshold is determined using only the training data
    """
    probas_pred = np.zeros(batch_loader.dataset.target_shape, dtype=np.float)
    y_true = batch_loader.dataset.targets[-len(probas_pred):]
    t_range = batch_loader.dataset.t_range[-len(probas_pred):]

    with torch.set_grad_enabled(False):
        model.eval()

        num_batches = batch_loader.num_batches
        for indices, spc_feats, tmp_feats, env_feats, targets in batch_loader:
            current_batch = batch_loader.current_batch

            # Transfer to PyTorch Tensor and GPU
            spc_feats = torch.Tensor(spc_feats[-1]).to(conf.device)  # only taking [0] for fnn
            tmp_feats = torch.Tensor(tmp_feats).to(conf.device)  # only taking [0] for fnn
            env_feats = torch.Tensor(env_feats[-1]).to(conf.device)  # only taking [0] for fnn
            # targets = torch.LongTensor(targets[0, :, 0]).to(conf.device)  # only taking [0] for fnn
            out = model(spc_feats, tmp_feats, env_feats)

            batch_probas_pred = F.softmax(out, dim=-1)[:, 1].cpu().numpy()  # select class1 prediction

            for i, p in zip(indices, batch_probas_pred):
                n, c, l = i
                probas_pred[n, c, l] = p

    return y_true, probas_pred, t_range
