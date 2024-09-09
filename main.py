# -*- coding:utf-8 -*-
import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import pickle as pkl
from apex import amp

import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from CENSOR import RobertaForTokenClassification_Modified
from utils.data_utils import load_and_cache_examples, get_labels
from utils.model_utils import mask_tokens, soft_frequency, get_uncertainty_aware_logits, update_pseudo_label_by_coteaching, opt_grad, get_hard_label, _update_mean_model_variables, _update_mean_model_variables_v2,_update_mean_model_variables_v3,_update_mean_model_variables_v4
from utils.eval import evaluate
from utils.config import config
from utils.loss_utils import NegEntropy

from models.m2_teacher import M2Teacher

import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

MODEL_NAMES = {
    "student1":"Roberta", 
    "student2":"DistilRoberta", 
    "teacher1":"Roberta", 
    "teacher2":"DistilRoberta"
}
MODEL_CLASSES = {
    "student1": (RobertaConfig, RobertaForTokenClassification_Modified, RobertaTokenizer),
    "student2": (RobertaConfig, RobertaForTokenClassification_Modified, RobertaTokenizer),
}
LOSS_WEIGHTS = {
    "pseudo": 1.0,
    "self": 0.5,
    "mutual": 0.3,
    "mean": 0.2,
}
torch.set_printoptions(profile="full")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, t_total, num_labels, epoch):
    config_class, model_class, _ = MODEL_CLASSES["student1"]
    config_s1 = config_class.from_pretrained(
        args.student1_config_name if args.student1_config_name else args.student1_model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_s1 = model_class.from_pretrained(
        args.student1_model_name_or_path,
        from_tf=bool(".ckpt" in args.student1_model_name_or_path),
        config=config_s1,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_s1.to(args.device)

    config_class, model_class, _ = MODEL_CLASSES["student2"]
    config_s2 = config_class.from_pretrained(
        args.student2_config_name if args.student2_config_name else args.student2_model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_s2 = model_class.from_pretrained(
        args.student2_model_name_or_path,
        from_tf=bool(".ckpt" in args.student2_model_name_or_path),
        config=config_s2,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_s2.to(args.device)

    config_class, model_class, _ = MODEL_CLASSES["student1"]
    config_t1 = config_class.from_pretrained(
        args.student1_config_name if args.student1_config_name else args.student1_model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_t1 = model_class.from_pretrained(
        args.student1_model_name_or_path,
        from_tf=bool(".ckpt" in args.student1_model_name_or_path),
        config=config_t1,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_t1.to(args.device)

    config_class, model_class, _ = MODEL_CLASSES["student2"]
    config_t2 = config_class.from_pretrained(
        args.student2_config_name if args.student2_config_name else args.student2_model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_t2 = model_class.from_pretrained(
        args.student2_model_name_or_path,
        from_tf=bool(".ckpt" in args.student2_model_name_or_path),
        config=config_t2,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model_t2.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters_1 = [
        {
            "params": [p for n, p in model_s1.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_s1.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_s1 = AdamW(optimizer_grouped_parameters_1, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1,args.adam_beta2))
    scheduler_s1 = get_linear_schedule_with_warmup(
        optimizer_s1, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    optimizer_grouped_parameters_2 = [
        {
            "params": [p for n, p in model_s2.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model_s2.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_s2 = AdamW(optimizer_grouped_parameters_2, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1,args.adam_beta2))
    scheduler_s2 = get_linear_schedule_with_warmup(
        optimizer_s2, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        [model_s1, model_s2, model_t1, model_t2], [optimizer_s1, optimizer_s2] = amp.initialize(
                     [model_s1, model_s2, model_t1, model_t2], [optimizer_s1, optimizer_s2], opt_level=args.fp16_opt_level)


    if args.n_gpu > 1:
        model_s1 = torch.nn.DataParallel(model_s1)
        model_s2 = torch.nn.DataParallel(model_s2)
        model_t1 = torch.nn.DataParallel(model_t1)
        model_t2 = torch.nn.DataParallel(model_t2)


    if args.local_rank != -1:
        model_s1 = torch.nn.parallel.DistributedDataParallel(
            model_s1, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        model_s2 = torch.nn.parallel.DistributedDataParallel(
            model_s2, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        model_t1 = torch.nn.parallel.DistributedDataParallel(
            model_t1, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
        model_t2 = torch.nn.parallel.DistributedDataParallel(
            model_t2, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    model_s1.zero_grad()
    model_s2.zero_grad()
    model_t1.zero_grad()
    model_t2.zero_grad()

    for param in model_t1.parameters():
        param.detach_()
    for param in model_t2.parameters():
        param.detach_()

    return model_s1, model_s2, model_t1, model_t2, optimizer_s1, scheduler_s1, optimizer_s2, scheduler_s2


def validation(args, model, tokenizer, labels, pad_token_label_id, best_dev, best_test,
               global_step, t_total, epoch, tors, writer=None):
    model_type = MODEL_NAMES[tors].lower()
    
    results, _, best_dev, is_updated1 = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_dev, mode="dev", \
        logger=logger, prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)

    if writer is not None:
        writer.add_scalar('loss/dev_loss/' + tors, results['loss'], global_step)
        writer.add_scalar('precision/dev_precision/' + tors, results['precision'], global_step)
        writer.add_scalar('f1/dev_f1/' + tors, results['f1'], global_step)
        writer.add_scalar('recall/dev_recall/' + tors, results['recall'], global_step)

    results, _, best_test, is_updated2 = evaluate(args, model, tokenizer, labels, pad_token_label_id, best_test, mode="test", \
        logger=logger, prefix='test [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)

    if writer is not None:
        writer.add_scalar('loss/test_loss/' + tors, results['loss'], global_step)
        writer.add_scalar('precision/test_precision/' + tors, results['precision'], global_step)
        writer.add_scalar('f1/test_f1/' + tors, results['f1'], global_step)
        writer.add_scalar('recall/test_recall/' + tors, results['recall'], global_step)

    if args.local_rank in [-1, 0] and is_updated1:
        path = os.path.join(args.output_dir + tors, "checkpoint-best-1")
        logger.info("Saving model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        ) 
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)
    if args.local_rank in [-1, 0] and is_updated2:
        path = os.path.join(args.output_dir + tors, "checkpoint-best-2")
        logger.info("Saving model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)

    return best_dev, best_test, is_updated1

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def random_sampler(args, label_, prob=None):
    label = copy.deepcopy(label_)
    mask = (label==0)
    non_entity = label[mask]
    size = non_entity.size(0)
    if prob is not None:
        prob_ = copy.deepcopy(prob)
        softmax = torch.nn.Softmax(dim=-1)
        prob_ = softmax(prob_)
        prob_ = prob_[mask].max(dim=-1)[0]
        prob_ = 1-prob_
    else:
        prob_ = torch.rand(size).to(args.device)
    num_samples = int(0.3*size)
    if num_samples <= 0:
        return label!=-100

    select_ids = torch.multinomial(prob_, num_samples)
    non_entity[select_ids] = -100
    label[label==0] = non_entity
    label_mask = (label!=-100)

    return label_mask
    
def initial_mask(args, batch):
    if args.dataset in []:
        return None, None
    else:
        label_mask1 = random_sampler(args, batch, prob=None)
        label_mask2 = random_sampler(args, batch, prob=None)
        return label_mask1, label_mask2

def get_teacher(args, model_t1, model_t2, t_model1, t_model2, dev_is_updated1, dev_is_updated2, batch=True):
    if args.dataset in ["conll03", "wikigold"] and batch:
        if dev_is_updated1:
            t_model1 = copy.deepcopy(model_t1)
        if dev_is_updated2:
            t_model2 = copy.deepcopy(model_t2)
    else:
        t_model1 = copy.deepcopy(model_t1)
        t_model2 = copy.deepcopy(model_t2)

    return t_model1, t_model2

def train(args, train_dataset, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    num_labels = len(labels)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps//(len(train_dataloader)//args.gradient_accumulation_steps)+1
    else:
        t_total = len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs

    model_s1, model_s2, model_t1, model_t2, optimizer_s1, scheduler_s1, optimizer_s2, scheduler_s2 = initialize(args, t_total, num_labels, 0)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)
    s1_best_dev, s1_best_test = [0, 0, 0], [0, 0, 0]
    s2_best_dev, s2_best_test = [0, 0, 0], [0, 0, 0]
    t1_best_dev, t1_best_test = [0, 0, 0], [0, 0, 0]
    t2_best_dev, t2_best_test = [0, 0, 0], [0, 0, 0]

    self_learning_teacher_model1 = model_s1
    self_learning_teacher_model2 = model_s2

    softmax = torch.nn.Softmax(dim=1)
    t_model1 = copy.deepcopy(model_s1)
    t_model2 = copy.deepcopy(model_s2)

    loss_regular = NegEntropy()

    writer = SummaryWriter(log_dir=os.path.join('./runs/', args.tensor_board_dir))

    begin_global_step = len(train_dataloader)*args.begin_epoch//args.gradient_accumulation_steps

    final_best_p = 0
    final_best_r = 0
    final_best_f = 0

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model_s1.train()
            model_s2.train()
            model_t1.train()
            model_t2.train()

            batch = tuple(t.to(args.device) for t in batch)
            module_list = nn.ModuleList([])
            TEST = False

            label_mask1, label_mask2 = initial_mask(args, batch[3])
            if epoch >= args.begin_epoch:
                delta = global_step - begin_global_step
                if delta//args.self_learning_period > 0:
                    if delta%args.self_learning_period == 0:
                        self_learning_teacher_model1 = copy.deepcopy(t_model1)
                        self_learning_teacher_model1.eval()
                        self_learning_teacher_model2 = copy.deepcopy(t_model2)
                        self_learning_teacher_model2.eval()


                    inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                    TEST = True 
                    with torch.no_grad():
                        outputs1 = self_learning_teacher_model1(**inputs)
                        outputs2 = self_learning_teacher_model2(**inputs)

                    pseudo_labels1 = torch.argmax(outputs1[0], axis=2)  
                    pseudo_labels2 = torch.argmax(outputs2[0], axis=2)  

                    with torch.no_grad():
                        ua_logits1_list = []
                        ua_logits2_list = []
                        num_uncertainty = 8 
                        ua_threshold = args.ua_threshold
                        for _ in range(num_uncertainty):
                            self_learning_teacher_model1.eval()
                            self_learning_teacher_model2.eval()
                            self_learning_teacher_model1.apply(apply_dropout)
                            self_learning_teacher_model2.apply(apply_dropout)
                            outputs1 = self_learning_teacher_model1(**inputs)
                            outputs2 = self_learning_teacher_model2(**inputs)
                            ua_logits1 = get_uncertainty_aware_logits(pseudo_labels1, outputs1[0], pad_token_label_id)
                            ua_logits2 = get_uncertainty_aware_logits(pseudo_labels2, outputs2[0], pad_token_label_id)
                            ua_logits1_list.append(ua_logits1)
                            ua_logits2_list.append(ua_logits2)
                        ua_logits1_list = torch.cat(ua_logits1_list, dim=-1)
                        ua_logits2_list = torch.cat(ua_logits2_list, dim=-1)
                        ua_logits1_var = torch.var(ua_logits1_list, dim=-1)
                        ua_logits2_var = torch.var(ua_logits2_list, dim=-1)
                        ua_logits1_var = F.softmax(ua_logits1_var,dim=-1)
                        ua_logits2_var = F.softmax(ua_logits2_var, dim=-1)

                        ua_mask1 = (ua_logits1_var <= ua_threshold)
                        ua_mask2 = (ua_logits2_var <= ua_threshold)

                else:
                    pseudo_labels1 = batch[3]
                    pseudo_labels2 = batch[3]

                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                with torch.no_grad():
                    outputs1 = t_model1(**inputs)
                    outputs2 = t_model2(**inputs)
                    module_list.append(t_model1)
                    module_list.append(t_model2)
                    logits1 = outputs1[0]
                    logits2 = outputs2[0]
                    pred_labels1 = torch.argmax(logits1, dim=-1)
                    pred_labels2 = torch.argmax(logits2, dim=-1)

                    label_mask1 = (pred_labels1==pseudo_labels2) & label_mask1
                    label_mask2 = (pred_labels2==pseudo_labels1) & label_mask2

                logits1 = soft_frequency(logits=logits1, power=2)
                logits2 = soft_frequency(logits=logits2, power=2)
                if args.self_learning_label_mode == "hard":
                    pred_labels1, label_mask1_ = mask_tokens(args, batch[3], pred_labels1, pad_token_label_id, pred_logits=logits1)
                    pred_labels2, label_mask2_ = mask_tokens(args, batch[3], pred_labels2, pad_token_label_id, pred_logits=logits2)
                elif args.self_learning_label_mode == "soft":
                    pred_labels1, label_mask1_ = mask_tokens(args, batch[3], logits1, pad_token_label_id)
                    pred_labels2, label_mask2_ = mask_tokens(args, batch[3], logits2, pad_token_label_id)


                if label_mask1_ is not None:
                    label_mask1 = label_mask1 & label_mask1_
                if label_mask2_ is not None:
                    label_mask2 = label_mask2 & label_mask2_

                coteach_epoch = (10 - delta//args.self_learning_period)
                if coteach_epoch < 0:
                    coteach_epoch = 0
                coteaching_prob = args.coteaching_prob
                with torch.no_grad():
                    outputs1 = model_s1(**inputs)
                    outputs2 = model_s2(**inputs)
                    new_pseudo_label2, coteach_mask_by_loss1 = update_pseudo_label_by_coteaching(pred_labels2, pred_labels1, outputs1[0], batch[1], coteaching_prob)
                    new_pseudo_label1, coteach_mask_by_loss2 = update_pseudo_label_by_coteaching(pred_labels1, pred_labels2, outputs2[0], batch[1], coteaching_prob)

                    pred_labels1 = new_pseudo_label1
                    pred_labels2 = new_pseudo_label2

                    COTEACH_TYPE = 0
                    if COTEACH_TYPE == 0:
                        label_mask1 = label_mask1 | coteach_mask_by_loss2
                        label_mask2 = label_mask2 | coteach_mask_by_loss1
            else:
                pred_labels1 = batch[3]
                pred_labels2 = batch[3]
                pseudo_labels1 = batch[3]
                pseudo_labels2 = batch[3]
                label_mask1, label_mask2 = initial_mask(args, batch[3]) 

                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                with torch.no_grad():

                    ua_logits1_list = []
                    ua_logits2_list = []
                    num_uncertainty = 8 
                    ua_threshold = args.ua_threshold
                    for _ in range(num_uncertainty):
                        model_s1.eval()
                        model_s2.eval()
                        model_s1.apply(apply_dropout)
                        model_s2.apply(apply_dropout)
                        outputs1 = model_s1(**inputs)
                        outputs2 = model_s2(**inputs)
                        ua_logits1 = get_uncertainty_aware_logits(pseudo_labels1, outputs1[0],
                                                                  pad_token_label_id) 
                        ua_logits2 = get_uncertainty_aware_logits(pseudo_labels2, outputs2[0], pad_token_label_id)
                        ua_logits1_list.append(ua_logits1)
                        ua_logits2_list.append(ua_logits2)
                    ua_logits1_list = torch.cat(ua_logits1_list, dim=-1)
                    ua_logits2_list = torch.cat(ua_logits2_list, dim=-1)
                    ua_logits1_var = torch.var(ua_logits1_list, dim=-1) 
                    ua_logits2_var = torch.var(ua_logits2_list, dim=-1)
                    ua_logits1_var = F.softmax(ua_logits1_var, dim=-1)
                    ua_logits2_var = F.softmax(ua_logits2_var, dim=-1)
                    ua_mask1 = (ua_logits1_var <= ua_threshold)
                    ua_mask2 = (ua_logits2_var <= ua_threshold)
                label_mask1 = label_mask1 & ua_mask1
                label_mask2 = label_mask2 & ua_mask2

                with torch.no_grad():
                    coteaching_prob = args.coteaching_prob
                    outputs1 = model_s1(**inputs)
                    outputs2 = model_s2(**inputs)
                    new_pseudo_label2, coteach_mask_by_loss1 = update_pseudo_label_by_coteaching(pred_labels2, pred_labels1, outputs1[0], batch[1], coteaching_prob)
                    new_pseudo_label1, coteach_mask_by_loss2 = update_pseudo_label_by_coteaching(pred_labels1, pred_labels2, outputs2[0], batch[1], coteaching_prob)

                    pred_labels1 = new_pseudo_label1
                    pred_labels2 = new_pseudo_label2

                label_mask1 = label_mask1 | coteach_mask_by_loss2
                label_mask2 = label_mask2 | coteach_mask_by_loss1

            model_s1.train()
            model_s2.train()
            inputs1 = {"input_ids": batch[0], "attention_mask": batch[1], "labels": {"pseudo": pred_labels1}, "label_mask": label_mask1, "module_list": module_list,"optimizer":optimizer_s1,"args":args,"TEST":TEST}
            outputs1 = model_s1(**inputs1)

            inputs2 = {"input_ids": batch[0], "attention_mask": batch[1], "labels": {"pseudo": pred_labels2}, "label_mask": label_mask2, "module_list": module_list,"optimizer":optimizer_s2,"args":args,"TEST":TEST}
            outputs2 = model_s2(**inputs2)        


            loss1 = 0.0
            loss_dict1 = outputs1[0]
            keys = loss_dict1.keys()
            for key in keys:
                loss1 += LOSS_WEIGHTS[key]*loss_dict1[key]
            loss2 = 0.0
            loss_dict2 = outputs2[0]
            keys = loss_dict2.keys()
            for key in keys:
                loss2 += LOSS_WEIGHTS[key]*loss_dict2[key]

            if args.n_gpu > 1:
                loss1 = loss1.mean()
                loss2 = loss2.mean()
            if args.gradient_accumulation_steps > 1:
                loss1 = loss1/args.gradient_accumulation_steps
                loss2 = loss2/args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss1, optimizer_s1) as scaled_loss1:
                    scaled_loss1.backward()
                with amp.scale_loss(loss2, optimizer_s2) as scaled_loss2:
                    scaled_loss2.backward()
            else:
                loss1.backward()
                loss2.backward()

            tr_loss += loss1.item()+loss2.item()

            if (step+1)%args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_s1), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_s2), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_s1.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(model_s2.parameters(), args.max_grad_norm)

                optimizer_s1.step()
                scheduler_s1.step()
                optimizer_s2.step()
                scheduler_s2.step() 
                model_s1.zero_grad()
                model_s2.zero_grad()
                global_step += 1

                _update_mean_model_variables_v3(model_s1, model_t1, args.mean_alpha, global_step,t_total,args.al)
                _update_mean_model_variables_v3(model_s2, model_t2, args.mean_alpha, global_step,t_total,args.al)

                writer.add_scalar('loss/train_loss', tr_loss/global_step, global_step)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step%args.logging_steps == 0:
                    if args.evaluate_during_training:
                        logger.info("***** Student1 combined Entropy loss : %.4f *****", loss1.item())
                        logger.info("##### Student1 #####")
                        s1_best_dev, s1_best_test, _ = validation(args, model_s1, tokenizer, labels, pad_token_label_id, \
                            s1_best_dev, s1_best_test, global_step, t_total, epoch, "student1")

                        logger.info("##### Teacher1 #####")
                        t1_best_dev, t1_best_test, dev_is_updated1 = validation(args, model_t1, tokenizer, labels, pad_token_label_id, \
                            t1_best_dev, t1_best_test, global_step, t_total, epoch, "teacher1")

                        logger.info("***** Student2 combined Entropy loss : %.4f *****", loss2.item())
                        logger.info("##### Student2 #####")
                        s2_best_dev, s2_best_test, _ = validation(args, model_s2, tokenizer, labels, pad_token_label_id, \
                            s2_best_dev, s2_best_test, global_step, t_total, epoch, "student2")
                        logger.info("##### Teacher2 #####")
                        t2_best_dev, t2_best_test, dev_is_updated2 = validation(args, model_t2, tokenizer, labels, pad_token_label_id, \
                            t2_best_dev, t2_best_test, global_step, t_total, epoch, "teacher2")

                        t_model1, t_model2 = get_teacher(args, model_t1, model_t2, t_model1, t_model2, dev_is_updated1, dev_is_updated2)


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        logger.info("***** Epoch : %d *****", epoch)
        logger.info("##### Student1 #####")
        s1_best_dev, s1_best_test, _  = validation(args, model_s1, tokenizer, labels, pad_token_label_id, \
            s1_best_dev, s1_best_test, global_step, t_total, epoch, "student1", writer)
        logger.info("##### Teacher1 #####")
        t1_best_dev, t1_best_test, dev_is_updated1 = validation(args, model_t1, tokenizer, labels, pad_token_label_id, \
            t1_best_dev, t1_best_test, global_step, t_total, epoch, "teacher1", writer)
        logger.info("##### Student2 #####")
        s2_best_dev, s2_best_test, _ = validation(args, model_s2, tokenizer, labels, pad_token_label_id, \
            s2_best_dev, s2_best_test, global_step, t_total, epoch, "student2")
        logger.info("##### Teacher2 #####")
        t2_best_dev, t2_best_test, dev_is_updated2 = validation(args, model_t2, tokenizer, labels, pad_token_label_id, \
            t2_best_dev, t2_best_test, global_step, t_total, epoch, "teacher2")

        t_model1, t_model2 = get_teacher(args, model_t1, model_t2, t_model1, t_model2, dev_is_updated1, dev_is_updated2, True)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    results = (t1_best_dev, t1_best_test, t2_best_dev, t2_best_test)

    return global_step, tr_loss/global_step, results

def main():
    args = config()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else: 
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", "%m/%d/%Y %H:%M:%S")
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    set_seed(args)
    labels = get_labels(args.data_dir, args.dataset)
    pad_token_label_id = CrossEntropyLoss().ignore_index

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss, best_results = train(args, train_dataset, tokenizer, labels, pad_token_label_id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
