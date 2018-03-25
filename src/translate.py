from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import _pickle as pickle
import shutil
import gc
import os
import sys
import time

import numpy as np

from data_utils_old import DataLoader
from hparams import *
from utils import *
from models_old import *

import torch
import torch.nn as nn
from torch.autograd import Variable

class TranslationHparams(HParams):
  dataset = "Translate dataset"

parser = argparse.ArgumentParser(description="Neural MT translator")

add_argument(parser, "cuda", type="bool", default=False, help="GPU or not")
add_argument(parser, "data_path", type="str", default=None, help="path to all data")
add_argument(parser, "model_dir", type="str", default="outputs", help="root directory of saved model")
add_argument(parser, "source_test", type="str", default=None, help="name of source test file")
add_argument(parser, "target_test", type="str", default=None, help="name of target test file")
add_argument(parser, "beam_size", type="int", default=None, help="beam size")
add_argument(parser, "max_len", type="int", default=300, help="maximum len considered on the target side")
add_argument(parser, "non_batch_translate", type="bool", default=False, help="use non-batched translation")
add_argument(parser, "batch_size", type="int", default=32, help="")
add_argument(parser, "merge_bpe", type="bool", default=True, help="")
add_argument(parser, "source_vocab", type="str", default=None, help="name of source vocab file")
add_argument(parser, "target_vocab", type="str", default=None, help="name of target vocab file")
add_argument(parser, "n_train_sents", type="int", default=None, help="max number of training sentences to load")
add_argument(parser, "out_file", type="str", default="trans", help="output file for hypothesis")


args = parser.parse_args()

model_file_name = os.path.join(args.model_dir, "model.pt")
model = torch.load(model_file_name)
model.eval()

out_file = os.path.join(args.model_dir, args.out_file)

hparams = TranslationHparams(
  data_path=args.data_path,
  source_vocab=args.source_vocab,
  target_vocab=args.target_vocab,
  source_test = args.source_test,
  target_test = args.target_test,
  cuda=args.cuda,
  beam_size=args.beam_size,
  max_len=args.max_len,
  batch_size=args.batch_size,
  n_train_sents=args.n_train_sents,
  merge_bpe=args.merge_bpe,
  out_file=out_file,
)

hparams.add_param("filtered_tokens", set([model.hparams.pad_id, model.hparams.eos_id, model.hparams.bos_id]))
model.hparams.cuda = hparams.cuda
data = DataLoader(hparams=hparams, decode=True)

out_file = open(hparams.out_file, 'w', encoding='utf-8')
end_of_epoch = False
num_sentences = 0

x_test = data.x_test.tolist()
hyps = model.translate(
      x_test, beam_size=args.beam_size, max_len=args.max_len)
#while not end_of_epoch:
#  ((x_test, x_mask, x_len, x_count),
#   (y_test, y_mask, y_len, y_count),
#   batch_size, end_of_epoch) = data.next_test(test_batch_size=hparams.batch_size)
#
#  num_sentences += batch_size
#
#  # The normal, correct way:
#  hyps = model.translate(
#        x_test, x_len, beam_size=args.beam_size, max_len=args.max_len)
#  # For debugging:
#  # model.debug_translate_batch(
#  #   x_test, x_mask, x_pos_emb_indices, hparams.beam_size, hparams.max_len,
#  #   y_test, y_mask, y_pos_emb_indices)
#  # sys.exit(0)

for h in hyps:
  #print(h)
  h_best_words = map(lambda wi: data.target_index_to_word[wi],
                     filter(lambda wi: wi not in hparams.filtered_tokens, h))
  if hparams.merge_bpe:
    line = ''.join(h_best_words)
    line = line.replace('▁', ' ')
  else:
    line = ' '.join(h_best_words)
  line = line.strip()
  out_file.write(line + '\n')
  out_file.flush()

print("Translated {0} sentences".format(num_sentences))
sys.stdout.flush()

out_file.close()

