import logging as log
import numpy as np
import torch
import torch.nn as nn
from utils.configs import BaseConf


def train_epoch_for_sequence_model(model, optimiser, batch_loader, loss_fn, total_losses, conf: BaseConf):
    batch_losses = []
    num_batches = len(batch_loader)

    for current_batch, (indices, inputs, targets) in enumerate(batch_loader):
        out = model(inputs)
        loss = loss_fn(input=out, target=targets)

        batch_losses.append(loss.item())
        total_losses.append(batch_losses[-1])

        if model.training:  # not used in validation loops
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            log.debug(f"Batch: {current_batch + 1:04d}/{num_batches:04d} \t Loss: {batch_losses[-1]:.4f}")

    epoch_loss = np.mean(batch_losses)
    return epoch_loss


def evaluate_sequence_model(model: nn.Module, batch_loader, conf: BaseConf):
    y_score = np.zeros(batch_loader.dataset.target_shape, dtype=np.float)
    y_count = np.zeros_like(y_score)

    with torch.set_grad_enabled(False):
        model.eval()

        for current_batch, (indices, inputs, targets) in enumerate(batch_loader):

            out = model(inputs)  # b,s,f

            #             batch_y_score = out[:, -1, 0].cpu().numpy()  # select class1 prediction
            #             batch_y_count = targets[:, -1, 0].cpu().numpy()  # select class1 prediction

            batch_y_score = out[:, -1, :].cpu().numpy()  # select class1 prediction
            batch_y_count = targets[:, -1, :].cpu().numpy()  # select class1 prediction

            for i, s, c in zip(indices, batch_y_score, batch_y_count):
                y_score[i] = s
                y_count[i] = c

    return y_count, y_score