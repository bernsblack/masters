import logging as log
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from utils.configs import BaseConf


def train_epoch_for_sequence_model(
        model: nn.Module,
        optimiser: Optimizer,
        batch_loader: DataLoader,
        loss_fn: Callable,
        total_losses: List[float],
        conf: BaseConf,
):
    batch_losses = []
    num_batches = len(batch_loader)

    for current_batch, (indices, _inputs, _targets) in enumerate(batch_loader):
        inputs, targets = _inputs.to(conf.device), _targets.to(conf.device)

        out = model(inputs)
        loss = loss_fn(input=out, target=targets)

        batch_losses.append(loss.cpu().item())
        total_losses.append(batch_losses[-1])

        if model.training:  # not used in validation loops
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            log.debug(f"Batch: {current_batch + 1:04d}/{num_batches:04d} \t Loss: {batch_losses[-1]:.4f}")

    epoch_loss = np.mean(batch_losses)
    return epoch_loss


def evaluate_sequence_model(
        model: nn.Module,
        batch_loader: DataLoader,
        conf: BaseConf,
) -> (ndarray, ndarray):
    y_score = np.zeros(batch_loader.dataset.target_shape, dtype=np.float)
    y_count = np.zeros_like(y_score)

    with torch.set_grad_enabled(False):
        model.eval()

        for current_batch, (indices, _inputs, _targets) in enumerate(batch_loader):
            inputs, targets = _inputs.to(conf.device), _targets.to(conf.device)

            out = model(inputs)  # b,s,f

            #             batch_y_score = out[:, -1, 0].cpu().numpy()  # select class1 prediction
            #             batch_y_count = targets[:, -1, 0].cpu().numpy()  # select class1 prediction

            # TODO: ensure we're only evaluating on the final forecast in the sequence to avoid train set leakage
            batch_y_score = out[:, -1, :].cpu().numpy()  # select class1 prediction
            batch_y_count = targets[:, -1, :].cpu().numpy()  # select class1 prediction

            for i, s, c in zip(indices, batch_y_score, batch_y_count):
                y_score[i] = s
                y_count[i] = c

    return y_count, y_score
