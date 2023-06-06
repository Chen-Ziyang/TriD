import torch
import numpy as np
import torch.nn as nn


def dice_coeff(pred, label):
    smooth = 1.
    bs = pred.size(0)
    m1 = pred.contiguous().view(bs, -1)
    m2 = label.contiguous().view(bs, -1)
    intersection = (m1 * m2).sum()
    score = 1 - ((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth))
    return score


def jaccard_loss(pred, label):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m1 = torch.abs(1 - m1)
    m2 = label.view(num, -1)  # Flatten
    m2 = torch.abs(1 - m2)
    score = 1 - ((torch.min(m1, m2).sum() + smooth) / (torch.max(m1, m2).sum() + smooth))
    return score


def p2p_loss(pred, label):
    # Mean Absolute Error (MAE)
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = label.view(num, -1)  # Flatten
    score = torch.mean(torch.abs(m2 - m1))
    return score


def bce_loss(pred, label):
    score = torch.nn.BCELoss()(pred, label)
    return score


def mse_loss(pred, label):
    score = torch.nn.MSELoss()(pred, label)
    return score


def tversky_index(pred, label, alpha=0.7):
    smooth = 1
    num = pred.size(0)
    y_true_pos = label.view(num, -1)
    y_pred_pos = pred.view(num, -1)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1. - y_pred_pos))
    false_pos = torch.sum((1. - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                1 - alpha) * false_pos + smooth)


def focal_tversky(pred, label, gamma=2, alpha=0.7):
    pt_1 = tversky_index(pred, label, alpha)
    score = (1. - pt_1).pow(gamma)
    return score


class Seg_loss(nn.Module):
    def __init__(self):
        super(Seg_loss, self).__init__()

    def forward(self, logit_pred, label):
        pred = torch.nn.Sigmoid()(logit_pred)
        score = dice_coeff(pred=pred, label=label) + bce_loss(pred=pred, label=label)
        return score


def entropy_loss_func(v, sigmoid=False):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    if sigmoid:
        v = torch.nn.Sigmoid()(v)
    n, c, h, w = v.size()
    loss = -torch.sum(torch.mul(v, torch.log2(v + 1e-6))) / (n * h * w * np.log2(c))
    return loss


class EpochLR(torch.optim.lr_scheduler._LRScheduler):
    # lr_n = lr_0 * (1 - epoch / epoch_nums)^gamma
    def __init__(self, optimizer, epochs, gamma=0.9, last_epoch=-1):
        self.lr = optimizer.param_groups[0]['lr']
        self.epochs = epochs
        self.gamma = gamma
        super(EpochLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr * pow((1. - (self.last_epoch + 1) / self.epochs), self.gamma)]

