import os
import sys
import time

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

def get_criterion(hparams):
  loss_reduce = True 
  if hparams.trdec:
    loss_reduce = False
  crit = nn.CrossEntropyLoss(ignore_index=hparams.pad_id, size_average=False, reduce=loss_reduce)
  if hparams.cuda:
    crit = crit.cuda()
  return crit

def get_performance(crit, logits, labels, hparams):
  mask = (labels == hparams.pad_id)
  loss = crit(logits, labels)
  _, preds = torch.max(logits, dim=1)
  acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum()
  if hparams.trdec:
    rule_mask = (labels >= hparams.target_word_vocab_size)
    eos_mask = (labels == hparams.eos_id)
    word_mask = (labels < hparams.target_word_vocab_size) ^ eos_mask ^ mask
    rule_loss = loss[rule_mask].sum()
    eos_loss = loss[eos_mask].sum()
    word_loss = loss[word_mask].sum()
    return loss.sum(), acc, rule_loss, word_loss, eos_loss, rule_mask.long().sum(), word_mask.long().sum(), eos_mask.long().sum()
  else:
    return loss.sum(), acc

def count_params(params):
  num_params = sum(p.data.nelement() for p in params)
  return num_params

def save_checkpoint(extra, model, optimizer, hparams, path):
  print("Saving model to '{0}'".format(path))
  torch.save(extra, os.path.join(path, "extra.pt"))
  torch.save(model, os.path.join(path, "model.pt"))
  torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
  torch.save(hparams, os.path.join(path, "hparams.pt"))

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

def set_lr(optim, lr):
  for param_group in optim.param_groups:
    param_group["lr"] = lr

def init_param(p, init_type="uniform", init_range=None):
  if init_type == "xavier_normal":
    init.xavier_normal(p)
  elif init_type == "xavier_uniform":
    init.xavier_uniform(p)
  elif init_type == "kaiming_normal":
    init.kaiming_normal(p)
  elif init_type == "kaiming_uniform":
    init.kaiming_uniform(p)
  elif init_type == "uniform":
    assert init_range is not None and init_range > 0
    init.uniform(p, -init_range, init_range)
  else:
    raise ValueError("Unknown init_type '{0}'".format(init_type))

def add_argument(parser, name, type, default, help):
  """Add an argument.

  Args:
    name: arg's name.
    type: must be ["bool", "int", "float", "str"].
    default: corresponding type of value.
    help: help message.
  """

  if type == "bool":
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument("--{0}".format(name), dest=name,
                                action="store_true", help=help)
    feature_parser.add_argument("--no_{0}".format(name), dest=name,
                                action="store_false", help=help)
    parser.set_defaults(name=default)
  elif type == "int":
    parser.add_argument("--{0}".format(name),
                        type=int, default=default, help=help)
  elif type == "float":
    parser.add_argument("--{0}".format(name),
                        type=float, default=default, help=help)
  elif type == "str":
    parser.add_argument("--{0}".format(name),
                        type=str, default=default, help=help)
  else:
    raise ValueError("Unknown type '{0}'".format(type))

