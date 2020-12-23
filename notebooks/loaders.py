import numpy as np
import torch
import os


def load_train_avg(path):
    ckpt = torch.load(path + '/model.pt')
    return torch.as_tensor(ckpt['avg_reward'])


def load_train_std(path):
    ckpt = torch.load(path + '/model.pt')
    return torch.as_tensor(ckpt['std_reward'])


def stats_train(method=None, source=None, test_type=None):
    path = '../output/' + method + '/Pendulum-' + source + '/Results-' + test_type
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    rewards = torch.cat(([load_train_avg(folder).unsqueeze(0) for folder in subfolders]), dim=0)
    stds = torch.cat(([load_train_std(folder).unsqueeze(0) for folder in subfolders]), dim=0)
    mean = torch.mean(rewards, dim=0)
    std = torch.mean(stds, dim=0)
    return mean, std


def load_test_avg(path):
    ckpt = torch.load(path + '/test_result.pt')
    return [np.average(ckpt)]


def load_test_std(path):
    ckpt = torch.load(path + '/test_result.pt')
    return [np.std(ckpt)]


def stats_test(method=None, source=None, test_type=None):
    path = '../output/' + method + '/Pendulum-' + source + '/Results-' + test_type
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    rewards = [load_train_avg(folder) for folder in subfolders]
    stds = [load_train_std(folder) for folder in subfolders]
    mean = np.average(rewards)
    std = np.average(stds)
    return mean, std
