import numpy as np
import torch
import shutil


def save_checkpoint(state, is_best, filename='models/checkpoint.pth.tar'):
    """
    state = {'model_arch': [{'n_channels': 6,
                    'n_layers': 12,
                    'kernel_size': 3,
                    'lc':3,
                    'lp':3,
                    'lq':3,
                    'c':1,
                    'p':24,
                    'c':168}],
                'epoch':0,
                'batch_size':32,
                'best_loss':np.inf,
                'learn_rate':0.001,
                'model_state_dict':[],
                'optimizer_state_dict':[],
                'time':int(time())}
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'models/model_best.pth.tar')


def load_checkpoint(load_best=False):
    if load_best:
        filename = 'models/model_best.pth.tar'
    else:
        filename = 'models/checkpoint.pth.tar'

    return torch.load(filename)


def get_n_params(model):
    """
    Returns the number of parameters the PyTorch model has
    """
    total = 0
    for i in list(model.parameters()):
        p = np.product(i.shape)
        total += p

    return total
