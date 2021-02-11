import numpy as np
import torch
import os


def load_train_avg(path, epoch=0):
    if epoch != 0:
        model = '/model_' + str(epoch) + '.pt'
    else:
        model = '/model.pt'
    ckpt = torch.load(path + model)
    return torch.as_tensor(ckpt['avg_reward'])


def load_train_std(path, epoch=0):
    if epoch != 0:
        model = '/model_' + str(epoch) + '.pt'
    else:
        model = '/model.pt'
    ckpt = torch.load(path + model)
    return torch.as_tensor(ckpt['std_reward'])


def stats_train(method=None, source=None, test_type=None, epoch=0):
    path = '../output/' + method + '/Pendulum-' + source + '/Results-' + test_type
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    rewards = torch.cat(([load_train_avg(folder, epoch).unsqueeze(0) for folder in subfolders]), dim=0)
    stds = torch.cat(([load_train_std(folder, epoch).unsqueeze(0) for folder in subfolders]), dim=0)
    mean = torch.mean(rewards, dim=0)
    std = torch.mean(stds, dim=0)
    return mean, std


def stats_train(method=None, source=None, test_type=None, epoch=0):
    path = '../output/' + method + '/Pendulum-' + source + '/Results-' + test_type
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    rewards = torch.cat(([load_train_avg(folder, epoch).unsqueeze(0) for folder in subfolders]), dim=0)
    stds = torch.cat(([load_train_std(folder, epoch).unsqueeze(0) for folder in subfolders]), dim=0)
    mean = torch.mean(rewards, dim=0)
    std = torch.mean(stds, dim=0)
    return mean, std


def load_test_avg(path, epoch=None):
    if epoch is not None:
        test = '/test_result_' + str(epoch) + '.pt'
    else:
        test = '/test_result.pt'
    ckpt = torch.load(path + test)
    return [np.average(ckpt['reward'])]


def load_test_std(path, epoch=None):
    if epoch is not None:
        test = '/test_result_' + str(epoch) + '.pt'
    else:
        test = '/test_result.pt'
    ckpt = torch.load(path + test)
    return [np.std(ckpt['reward'])]


def stats_test(method=None, source=None, test_type=None, epoch=None):
    path = '../output/' + method + '/Pendulum-' + source + '/Results-' + test_type
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    rewards = [load_test_avg(folder, epoch=epoch) for folder in subfolders]
    stds = [load_test_std(folder, epoch=epoch) for folder in subfolders]
    mean = np.average(rewards)
    std = np.average(stds)
    return mean, std
