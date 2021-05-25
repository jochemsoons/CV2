import os
import imageio
import numpy as np
import torch
import torch.nn as nn


def saveModels(model, optims, iterations, path):
    # cpu = torch.device('cpu')
    if isinstance(model, nn.DataParallel):
        checkpoint = {
            'iters': iterations,
            'model': model.module.state_dict(),
            'optimizer': optims.state_dict()
        }
    else:
        checkpoint = {
            'iters': iterations,
            'model': model.state_dict(),
            'optimizer': optims.state_dict()
        }
    torch.save(checkpoint, path)

def loadModels(model, path, optims=None, Test=True):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    if not Test:
        optims.load_state_dict(checkpoint['optimizer'])
    return model, optims, checkpoint['iters']
