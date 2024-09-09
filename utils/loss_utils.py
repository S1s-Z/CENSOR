# -*- coding:utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *

class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes, epsilon=0.1):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()
		return loss

class SoftEntropy(nn.Module):
	def __init__(self):
		super(SoftEntropy, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
		return loss

class NegEntropy(nn.Module):
	def __init__(self):
		super(NegEntropy, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs):
		log_probs = self.logsoftmax(inputs)
		loss = (F.softmax(inputs, dim=1) * log_probs).mean(0).sum()
		return loss


class GCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=0, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class his_GCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        valid_idx = targets != -100
        logits = logits[valid_idx]
        targets = targets[valid_idx]
        pred = F.softmax(logits, dim=-1)
        pred = torch.gather(pred, dim=-1, index=torch.unsqueeze(targets, -1))
        loss = (1-pred**self.q) / self.q
        loss = (loss.view(-1)).sum() / loss.shape[0]
        return loss.long()

class FocalLoss(nn.Module):
	def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.eps = eps
		self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

	def forward(self, inputs, targets):
		log_probs = self.ce(inputs, targets)
		probs = torch.exp(-log_probs)
		loss = (1-probs)**self.gamma*log_probs
		return loss

class SoftFocalLoss(nn.Module):
	def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
		super(SoftFocalLoss, self).__init__()
		self.gamma = gamma
		self.eps = eps
		self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		loss = (- F.softmax(targets, dim=1).detach() * log_probs * ((1-F.softmax(inputs, dim=1))**self.gamma)).mean(0).sum()
		return loss



class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        
        if isinstance(y_t, list):
            p_t_list = [F.softmax(y/self.T, dim=1) for y in y_t]
            p_t_tensor = torch.stack(p_t_list)
            p_t = p_t_tensor.mean(0)
        else:
            p_t = F.softmax(y_t/self.T, dim=1)

        loss = F.kl_div(p_s, p_t, reduction='sum') * \
            (self.T**2) / y_s.shape[0]
        return loss

