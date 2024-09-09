# -*- coding:utf-8 -*-
import logging
import os
import json
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn import CrossEntropyLoss, KLDivLoss
import copy

logger = logging.getLogger(__name__)

def soft_frequency(logits, power=2, probs=False):
    """
    Unsupervised Deep Embedding for Clustering Analysiszaodian
    https://arxiv.org/abs/1511.06335
    """
    if not probs:
        softmax = torch.nn.Softmax(dim=1)
        y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
    else:
        y = logits
    f = torch.sum(y, dim=(0, 1))
    t = y**power / f
    p = t/torch.sum(t, dim=2, keepdim=True)

    return p

def get_hard_label(args, combined_labels, pred_labels, pad_token_label_id, pred_logits=None):
    pred_labels[combined_labels==pad_token_label_id] = pad_token_label_id

    return pred_labels, None

def mask_tokens(args, combined_labels, pred_labels, pad_token_label_id, pred_logits=None):

    if args.self_learning_label_mode == "hard":
        softmax = torch.nn.Softmax(dim=1)
        y = softmax(pred_logits.view(-1, pred_logits.shape[-1])).view(pred_logits.shape)
        _threshold = args.threshold
        pred_labels[y.max(dim=-1)[0]>_threshold] = pad_token_label_id
        return pred_labels, None

    elif args.self_learning_label_mode == "soft":
        label_mask = (pred_labels.max(dim=-1)[0]>args.threshold)

        
        return pred_labels, label_mask

def opt_grad(loss, in_var, optimizer):
    
    if hasattr(optimizer, 'scalar'):
        loss = loss * optimizer.scaler.loss_scale
    return torch.autograd.grad(loss, in_var)

def _update_mean_model_variables(model, m_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for m_param, param in zip(m_model.parameters(), model.parameters()):
        m_param.data.mul_(alpha).add_(1 - alpha, param.data)   


def _update_mean_model_variables_v2(stu_model, teach_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
        tmp_prob = np.random.rand()
        if tmp_prob < 0.7:
            pass
        else:
            p2.data = 0.99 * p2.data + (1.0 - 0.99) * p1.detach().data


def _update_mean_model_variables_v3(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    m = get_param_momentum(param_momentum,global_step,t_total)
    for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
        tmp_prob = np.random.rand()
        if tmp_prob < 0.8:
            pass
        else:
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data

def get_param_momentum(param_momentum,current_train_iter,total_iters):

    return 1.0 - (1.0 - param_momentum) * (
        (math.cos(math.pi * current_train_iter / total_iters) + 1) * 0.5
    )

def _update_mean_prediction_variables(prediction, m_prediction, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    m_prediction.data.mul_(alpha).add_(1 - alpha, prediction.data)

def _update_mean_model_variables_v4(stu_model, teach_model, alpha, global_step,t_total,param_momentum):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    m = get_param_momentum(param_momentum,global_step,t_total)
    layer = 1
    temp1 = []
    temp2 = []
    start = 182
    end = 197
    for p1, p2 in zip(stu_model.parameters(), teach_model.parameters()):    
        if start <= layer <= end:
            temp1.append(p1)
            temp2.append(p2)
            if layer == end :
                sts(temp1,temp2,m)
                temp1 = []
                temp2 = []
        else :
            p2.data.mul_(alpha).add_(1 - alpha, p1.data)
        layer += 1
def sts(temp1,temp2,m):
    tmp_prob = np.random.rand()
    if tmp_prob < 0.8:
        pass
    else:
        for tmp1, tmp2 in zip(temp1,temp2):
            tmp2.data = m * tmp2.data + (1.0 - m) * tmp1.detach().data

def get_uncertainty_aware_logits(pseudo_labels, logits, pad_token_label_id):
    pseudo_index = torch.unsqueeze(pseudo_labels, dim=-1)
    pseudo_index[pseudo_index == pad_token_label_id] = 0
    ua_logits = torch.gather(logits, dim=-1, index=pseudo_index)
    return ua_logits


def update_pseudo_label_by_coteaching(old_label, ref_label, logits, attention_mask, coteaching_prob=0.2):
    def get_loss_from_logits(logits, labels):

        if labels.shape == logits.shape:
            loss_fct = KLDivLoss(reduction='none')
            loss = loss_fct(logits, labels).mean(dim=-1)
        else:
            logits = logits.transpose(1,2)
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits, labels)
        
        return loss

    new_label = copy.deepcopy(old_label)

    loss = get_loss_from_logits(logits, ref_label) 
    ref_mask = attention_mask==1 
    valid_count = ref_mask.sum(dim=-1)
    num_coteaching_select = (valid_count.float() * coteaching_prob).ceil().long()
    
    coteaching_mask = torch.zeros_like(loss, dtype=torch.bool)
    for i in range(loss.shape[0]):
        if num_coteaching_select[i] > 0:
            _, indices = torch.topk(loss[i][ref_mask[i]], num_coteaching_select[i], largest=False)
            coteaching_mask[i][ref_mask[i].nonzero(as_tuple=True)[0][indices]] = True
            new_label[i][ref_mask[i].nonzero(as_tuple=True)[0][indices]] = ref_label[i][ref_mask[i].nonzero(as_tuple=True)[0][indices]] 
    
    return new_label, coteaching_mask
    
    