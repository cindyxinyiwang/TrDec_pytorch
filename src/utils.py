import os
import sys
import time

from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

def get_criterion(hparams):
  crit = nn.CrossEntropyLoss(ignore_index=hparams.pad_id, size_average=False)
  if hparams.cuda:
    crit = crit.cuda()
  return crit

def get_performance(crit, logits, labels, hparams):
  mask = (labels == hparams.pad_id)
  loss = crit(logits, labels)
  _, preds = torch.max(logits, dim=1)
  acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum()

  return loss, acc

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
