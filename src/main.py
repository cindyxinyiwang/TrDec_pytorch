import numpy as np
import argparse
import time
import shutil
import gc
import random
import subprocess
import re

import torch
import torch.nn as nn
from torch.autograd import Variable

from data_utils import DataLoader
from hparams import *
from utils import *
from models import *
from trdec import *

parser = argparse.ArgumentParser(description="Neural MT")

parser.add_argument("--trdec", action="store_true", help="use the trdec model")

parser.add_argument("--load_model", action="store_true", help="load an existing model")
parser.add_argument("--reset_output_dir", action="store_true", help="delete output directory if it exists")
parser.add_argument("--output_dir", type=str, default="outputs", help="path to output directory")
parser.add_argument("--log_every", type=int, default=50, help="how many steps to write log")
parser.add_argument("--eval_every", type=int, default=500, help="how many steps to compute valid ppl")
parser.add_argument("--clean_mem_every", type=int, default=10, help="how many steps to clean memory")
parser.add_argument("--eval_bleu", action="store_true", help="if calculate BLEU score for dev set")
parser.add_argument("--beam_size", type=int, default=5, help="beam size for dev BLEU")

parser.add_argument("--cuda", action="store_true", help="GPU or not")

parser.add_argument("--max_len", type=int, default=300, help="maximum len considered on the target side")
parser.add_argument("--n_train_sents", type=int, default=None, help="max number of training sentences to load")

parser.add_argument("--d_word_vec", type=int, default=288, help="size of word and positional embeddings")
parser.add_argument("--d_model", type=int, default=288, help="size of hidden states")

parser.add_argument("--data_path", type=str, default=None, help="path to all data")
parser.add_argument("--source_train", type=str, default=None, help="source train file")
parser.add_argument("--target_train", type=str, default=None, help="target train file")
parser.add_argument("--source_valid", type=str, default=None, help="source valid file")
parser.add_argument("--target_valid", type=str, default=None, help="target valid file")
parser.add_argument("--target_valid_ref", type=str, default=None, help="target valid file for reference")
parser.add_argument("--source_vocab", type=str, default=None, help="source vocab file")
parser.add_argument("--target_vocab", type=str, default=None, help="target vocab file")
parser.add_argument("--source_test", type=str, default=None, help="source test file")
parser.add_argument("--target_test", type=str, default=None, help="target test file")

parser.add_argument("--target_tree_train", type=str, default=None, help="target train tree file")
parser.add_argument("--target_tree_valid", type=str, default=None, help="target valid tree file")
parser.add_argument("--target_tree_test", type=str, default=None, help="target test file")
parser.add_argument("--target_tree_vocab", type=str, default=None, help="target rule vocab file")
parser.add_argument("--target_word_vocab", type=str, default=None, help="target word vocab file")

parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--batcher", type=str, default="sent", help="sent|word. Batch either by number of words or number of sentences")
parser.add_argument("--n_train_steps", type=int, default=100000, help="n_train_steps")
parser.add_argument("--dropout", type=float, default=0.1, help="probability of dropping")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_dec", type=float, default=0.5, help="learning rate decay")
parser.add_argument("--clip_grad", type=float, default=5., help="gradient clipping")
parser.add_argument("--patience", type=int, default=-1, help="patience")

parser.add_argument("--seed", type=int, default=19920206, help="random seed")

parser.add_argument("--init_range", type=float, default=0.1, help="L2 init range")
parser.add_argument("--init_type", type=str, default="uniform", help="uniform|xavier_uniform|xavier_normal|kaiming_uniform|kaiming_normal")

args = parser.parse_args()
def eval(model, data, crit, step, hparams, eval_bleu=False,
         valid_batch_size=20):
  valid_batch_size = 2
  print("Eval at step {0}. valid_batch_size={1}".format(step, valid_batch_size))

  model.eval()
  data.reset_valid()
  valid_words = 0
  valid_loss = 0
  valid_acc = 0
  n_batches = 0

  valid_total = valid_rule_count = valid_word_count = valid_eos_count = 0
  valid_word_loss, valid_rule_loss, valid_eos_loss = 0, 0, 0
  valid_bleu = None
  if eval_bleu:
    valid_hyp_file = os.path.join(args.output_dir, "dev.trans_{0}".format(step))
    out_file = open(valid_hyp_file, 'w', encoding='utf-8')
  while True:
    # clear GPU memory
    #gc.collect()

    # next batch
    ((x_valid, x_mask, x_len, x_count),
     (y_valid, y_mask, y_len, y_count),
     batch_size, end_of_epoch) = data.next_valid(valid_batch_size=valid_batch_size)
    #print(x_valid)
    #print(x_mask)
    #print(y_valid)
    #print(y_mask)
    # do this since you shift y_valid[:, 1:] and y_valid[:, :-1]
    if args.trdec:
      y_total_count, y_rule_count, y_word_count, y_eos_count = y_count
      valid_total += (y_total_count - batch_size)
      valid_rule_count += y_rule_count
      valid_word_count += (y_word_count - batch_size)
      valid_eos_count += y_eos_count
    else:
      y_count -= batch_size
      # word count
      valid_words += y_count

    if args.trdec:
      logits = model.forward(
        x_valid, x_mask, x_len,
        y_valid[:,:-1,:], y_mask[:,:-1], y_len, y_valid[:,1:,2])
      logits = logits.view(-1, hparams.target_word_vocab_size+hparams.target_rule_vocab_size)
      labels = y_valid[:,1:,0].contiguous().view(-1)
      val_loss, val_acc, rule_loss, word_loss, eos_loss, rule_count, word_count, eos_count =  \
        get_performance(crit, logits, labels, hparams)
      valid_word_loss += word_loss.data[0]
      valid_rule_loss += rule_loss.data[0]
      valid_eos_loss += eos_loss.data[0]
    else:
      logits = model.forward(
        x_valid, x_mask, x_len,
        y_valid[:,:-1], y_mask[:,:-1], y_len)
      logits = logits.view(-1, hparams.target_vocab_size)
      labels = y_valid[:,1:].contiguous().view(-1)
      val_loss, val_acc = get_performance(crit, logits, labels, hparams)
    n_batches += 1
    valid_loss += val_loss.data[0]
    valid_acc += val_acc.data[0]
    # print("{0:<5d} / {1:<5d}".format(val_acc.data[0], y_count))
    if end_of_epoch:
      break
  # BLEU eval
  if eval_bleu:
    x_valid = data.x_valid.tolist()
    #print(x_valid)
    #x_valid = Variable(torch.LongTensor(x_valid), volatile=True)
    hyps = model.translate(
      x_valid, beam_size=args.beam_size, max_len=args.max_len)
    for h in hyps:
      h_best_words = map(lambda wi: data.target_index_to_word[wi], h)
      line = ''.join(h_best_words)
      line = line.replace('▁', ' ').strip()
      out_file.write(line + '\n')

  if args.trdec:
    val_ppl = np.exp(valid_loss / valid_word_count)
    log_string = "val_step={0:<6d}".format(step)
    log_string += " loss={0:<6.2f}".format(valid_loss / valid_word_count)
    log_string += " acc={0:<5.4f}".format(valid_acc / valid_total)
    log_string += " val_ppl={0:<.2f}".format(val_ppl)
    log_string += " num_word={} num_rule={} num_eos={}".format(valid_word_count, valid_rule_count, valid_eos_count)
    log_string += " ppl_word={0:<8.2f}".format(np.exp(valid_word_loss / valid_word_count))
    log_string += " ppl_rule={0:<8.2f}".format(np.exp(valid_rule_loss / valid_rule_count))
    log_string += " ppl_eos={0:<8.2f}".format(np.exp(valid_eos_loss / valid_eos_count))
  else:
    val_ppl = np.exp(valid_loss / valid_words)
    log_string = "val_step={0:<6d}".format(step)
    log_string += " loss={0:<6.2f}".format(valid_loss / valid_words)
    log_string += " acc={0:<5.4f}".format(valid_acc / valid_words)
    log_string += " val_ppl={0:<.2f}".format(val_ppl)
  if eval_bleu:
    out_file.close()
    if args.target_valid_ref:
      ref_file = os.path.join(hparams.data_path, args.target_valid_ref)
    else:
      ref_file = os.path.join(hparams.data_path, args.target_valid)
    bleu_str = subprocess.getoutput(
      "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_file))
    log_string += "\n{}".format(bleu_str)
    bleu_str = bleu_str.split('\n')[-1].strip()
    reg = re.compile("BLEU = ([^,]*).*")
    valid_bleu = float(reg.match(bleu_str).group(1))
    log_string += " val_bleu={0:<.2f}".format(valid_bleu)
  print(log_string)
  model.train()
  #exit(0)
  return val_ppl, valid_bleu

def train():
  if args.load_model:
    hparams_file_name = os.path.join(args.output_dir, "hparams.pt")
    hparams = torch.load(hparams_file_name)
  else:
    hparams = HParams(
      data_path=args.data_path,
      source_train=args.source_train,
      target_train=args.target_train,
      source_valid=args.source_valid,
      target_valid=args.target_valid,
      source_vocab=args.source_vocab,
      target_vocab=args.target_vocab,
      source_test=args.source_test,
      target_test=args.target_test,
      max_len=args.max_len,
      n_train_sents=args.n_train_sents,
      cuda=args.cuda,
      d_word_vec=args.d_word_vec,
      d_model=args.d_model,
      batch_size=args.batch_size,
      batcher=args.batcher,
      n_train_steps=args.n_train_steps,
      dropout=args.dropout,
      lr=args.lr,
      init_type=args.init_type,
      init_range=args.init_range,
      trdec=args.trdec,
      target_tree_vocab=args.target_tree_vocab,
      target_word_vocab=args.target_word_vocab,
      target_tree_train=args.target_tree_train,
      target_tree_valid=args.target_tree_valid,
      target_tree_test=args.target_tree_test,
    )
  data = DataLoader(hparams=hparams)
  hparams.add_param("source_vocab_size", data.source_vocab_size)
  if args.trdec:
    hparams.add_param("target_rule_vocab_size", data.target_rule_vocab_size)
    hparams.add_param("target_word_vocab_size", data.target_word_vocab_size)
  else:
    hparams.add_param("target_vocab_size", data.target_vocab_size)
  hparams.add_param("pad_id", data.pad_id)
  hparams.add_param("unk_id", data.unk_id)
  hparams.add_param("bos_id", data.bos_id)
  hparams.add_param("eos_id", data.eos_id)
  hparams.add_param("n_train_steps", args.n_train_steps)

  # build or load model model
  print("-" * 80)
  print("Creating model")
  if args.load_model:
    model_file_name = os.path.join(args.output_dir, "model.pt")
    print("Loading model from '{0}'".format(model_file_name))
    model = torch.load(model_file_name)

    optim_file_name = os.path.join(args.output_dir, "optimizer.pt")
    print("Loading optimizer from {}".format(optim_file_name))
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr)
    optimizer_state = torch.load(optim_file_name)
    optim.load_state_dict(optimizer_state)

    extra_file_name = os.path.join(args.output_dir, "extra.pt")
    step, best_val_ppl, best_val_bleu, cur_attempt, lr = torch.load(extra_file_name)
  else:
    if args.trdec:
      model = TrDec(hparams=hparams)
    else:
      model = Seq2Seq(hparams=hparams)
    if args.init_type == "uniform":
      print("initialize uniform with range {}".format(args.init_range))
      for p in model.parameters():
        p.data.uniform_(-args.init_range, args.init_range)
    trainable_params = [
      p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable_params, lr=hparams.lr)
    #optim = torch.optim.Adam(trainable_params)

    step = 0
    best_val_ppl = 1e10
    best_val_bleu = 0
    cur_attempt = 0
    lr = hparams.lr

  crit = get_criterion(hparams)
  trainable_params = [
    p for p in model.parameters() if p.requires_grad]
  num_params = count_params(trainable_params)
  print("Model has {0} params".format(num_params))

  print("-" * 80)
  print("start training...")
  start_time = log_start_time = time.time()
  target_words, total_loss, total_corrects = 0, 0, 0
  target_rules, target_total, target_eos = 0, 0, 0
  total_word_loss, total_rule_loss, total_eos_loss = 0, 0, 0
  model.train()
  while True:
    ((x_train, x_mask, x_len, x_count),
     (y_train, y_mask, y_len, y_count),
     batch_size) = data.next_train()
    #print(x_train)
    #print(x_mask)
    #print(y_train)
    #print(y_mask)
    #exit(0)
    optim.zero_grad()
    if args.trdec:
      y_total_count, y_rule_count, y_word_count, y_eos_count = y_count
      target_total += (y_total_count - batch_size)
      target_rules += y_rule_count
      target_eos += y_eos_count
      target_words += (y_word_count - batch_size)

      logits = model.forward(x_train, x_mask, x_len, y_train[:,:-1,:], y_mask[:,:-1], y_len, y_train[:,1:,2])
      logits = logits.view(-1, hparams.target_word_vocab_size+hparams.target_rule_vocab_size)
      labels = y_train[:,1:,0].contiguous().view(-1)
      tr_loss, tr_acc, rule_loss, word_loss, eos_loss, rule_count, word_count, eos_count = \
        get_performance(crit, logits, labels, hparams)
      #print(y_rule_count)
      #print(rule_count.data[0])
      assert y_rule_count == rule_count.data[0], "data rule count {}, performance rule count {}".format(y_rule_count, rule_count.data[0])
      assert y_eos_count == eos_count.data[0], "data eos count {}, performance eos count {}".format(y_eos_count, eos_count.data[0])
      assert y_word_count - batch_size == word_count.data[0], "data word count {}, performance word count {}".format(y_word_count-batch_size, word_count.data[0])
      total_word_loss += word_loss.data[0]
      total_rule_loss += rule_loss.data[0]
      total_eos_loss += eos_loss.data[0]
    else:
      target_words += (y_count - batch_size)

      logits = model.forward(x_train, x_mask, x_len, y_train[:,:-1], y_mask[:,:-1], y_len)
      logits = logits.view(-1, hparams.target_vocab_size)
      labels = y_train[:,1:].contiguous().view(-1)
      tr_loss, tr_acc = get_performance(crit, logits, labels, hparams)
    total_loss += tr_loss.data[0]
    total_corrects += tr_acc.data[0]
    step += 1

    tr_loss.div_(batch_size).backward()
    grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
    #grad_norm = 0
    optim.step()

    if step % args.log_every == 0:
      epoch = step // data.n_train_batches
      curr_time = time.time()
      since_start = (curr_time - start_time) / 60.0
      elapsed = (curr_time - log_start_time) / 60.0
      log_string = "ep={0:<3d}".format(epoch)
      log_string += " steps={0:<6.2f}".format(step / 1000)
      log_string += " lr={0:<9.7f}".format(lr)
      log_string += " loss={0:<7.2f}".format(tr_loss.data[0])
      log_string += " |g|={0:<5.2f}".format(grad_norm)
      if args.trdec:
        log_string += " num_word={} num_rule={} num_eos={}".format(target_words, target_rules, target_eos)
        log_string += " ppl={0:<8.2f}".format(np.exp(total_loss / target_words))
        log_string += " ppl_word={0:<8.2f}".format(np.exp(total_word_loss / target_words))
        log_string += " ppl_rule={0:<8.2f}".format(np.exp(total_rule_loss / target_rules))
        log_string += " ppl_eos={0:<8.2f}".format(np.exp(total_eos_loss / target_eos))
        log_string += " acc={0:<5.4f}".format(total_corrects / target_total)
      else:
        log_string += " ppl={0:<8.2f}".format(np.exp(total_loss / target_words))
        log_string += " acc={0:<5.4f}".format(total_corrects / target_words)

      log_string += " wpm(k)={0:<5.2f}".format(target_words / (1000 * elapsed))
      log_string += " time(min)={0:<5.2f}".format(since_start)
      print(log_string)
    if step % args.eval_every == 0:
      val_ppl, val_bleu = eval(model, data, crit, step, hparams, eval_bleu=args.eval_bleu, valid_batch_size=20)	
      based_on_bleu = args.eval_bleu
      if based_on_bleu:
        if best_val_bleu <= val_bleu:
          save = True 
          best_val_bleu = val_bleu
          cur_attempt = 0
        else:
          save = False
          cur_attempt += 1
      else:
      	if best_val_ppl >= val_ppl:
          save = True
          best_val_ppl = val_ppl
          cur_attempt = 0 
      	else:
          save = False
          cur_attempt += 1
      if save:
      	save_checkpoint([step, best_val_ppl, best_val_bleu, cur_attempt, lr], 
      		             model, optim, hparams, args.output_dir)
      else:
        lr = lr * args.lr_dec
        set_lr(optim, lr)
      # reset counter after eval
      log_start_time = time.time()
      target_words = total_corrects = total_loss = 0
      target_rules = target_total = target_eos = 0
      total_word_loss = total_rule_loss = total_eos_loss = 0
    if args.patience >= 0:
      if cur_attempt > args.patience: break
    else:
      if step > args.n_train_steps: break 

def main():
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  if not os.path.isdir(args.output_dir):
    print("-" * 80)
    print("Path {} does not exist. Creating.".format(args.output_dir))
    os.makedirs(args.output_dir)
  elif args.reset_output_dir:
    print("-" * 80)
    print("Path {} exists. Remove and remake.".format(args.output_dir))
    shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

  print("-" * 80)
  log_file = os.path.join(args.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)
  train()

if __name__ == "__main__":
  main()



