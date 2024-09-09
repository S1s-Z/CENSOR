# -*- coding:utf-8 -*-
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, KLDivLoss, NLLLoss
from utils.loss_utils import GCELoss, DistillKL
from optimization import find_optimal_svm

import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}





def _generate_relative_positions_embeddings(depth, length=128, max_relative_position=127):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(length)
    range_mat = range_vec.repeat(length).view(length, length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position
    embeddings_table = np.zeros([vocab_size, depth])
    for pos in range(vocab_size):
        for i in range(depth // 2):
            embeddings_table[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / depth))
            embeddings_table[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / depth))

    embeddings_table_tensor = torch.tensor(embeddings_table).float()
    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                    num_classes=vocab_size).float()
    embeddings = torch.matmul(one_hot_relative_positions_matrix, embeddings_table_tensor)
    my_shape = list(final_mat.size())
    my_shape.append(depth)
    embeddings = embeddings.view(my_shape)
    # print(embeddings)
    return embeddings

class RobertaForTokenClassification_Modified(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            label_mask=None,
            module_list=None,
            optimizer=None,
            args=None,
            TEST=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        final_embedding = outputs[0]
        sequence_output = self.dropout(final_embedding)
        logits = self.classifier(sequence_output)

        outputs = (logits, final_embedding,) + outputs[2:]  # add hidden states and attention if they are here

        loss_dict = {}

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels is not None:
            active_loss = True
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1

            for key in labels:
                label = labels[key]
                if label is None:
                    continue
                if label_mask is not None:
                    all_active_loss = active_loss & label_mask.view(-1)
                else:
                    all_active_loss = active_loss
                active_logits = logits.view(-1, self.num_labels)[all_active_loss]

                if label.shape == logits.shape:
                    loss_fct = KLDivLoss()
                    if attention_mask is not None or label_mask is not None:
                        active_labels = label.view(-1, self.num_labels)[all_active_loss]
                        loss_ce = loss_fct(active_logits, active_labels)
                    else:
                        loss_ce = loss_fct(logits, label)
                else:
                    loss_fct = CrossEntropyLoss()

                    if attention_mask is not None or label_mask is not None:
                        active_labels = label.view(-1)[all_active_loss]
                        loss_ce = loss_fct(active_logits, active_labels)
                    else:
                        loss_ce = loss_fct(logits.view(-1, self.num_labels), label.view(-1))


                if module_list != None:
                    if len(module_list) == 2 and TEST:
                        deactive_logits = logits.view(-1, self.num_labels)[all_deactive_loss]
                        for module in module_list:
                            module.eval()
                        logit_t_list = []
                        with torch.no_grad():
                            for model_t in module_list:
                                outputs = model_t(**inputs)
                                logit_t = outputs[0].view(-1, self.num_labels)[all_active_loss]
                                logit_t_list.append(logit_t)
                        criterion_kd = DistillKL(2)
                        logit_s = active_logits
                        loss_div_list = []
                        grads = []
                        logit_s.register_hook(lambda grad: grads.append(
                            Variable(grad.data.clone(), requires_grad=False)))

                        optimizer.zero_grad()
                        logit_t_co = (logit_t_list[0] + logit_t_list[1]) / 2
                        logit_t_co = logit_t_co.to(args.device)
                        loss_s = criterion_kd(logit_s, logit_t_co)
                        loss_s.backward(retain_graph=True)
                        loss_div = loss_s


                if module_list != None:
                    if len(module_list) == 2 and TEST:
                        loss_dict[key] = args.bate * loss_div
                    else:
                        loss_dict[key] = loss_ce
                else:
                    loss_dict[key] = loss_ce

            outputs = (loss_dict,) + outputs

        return outputs
