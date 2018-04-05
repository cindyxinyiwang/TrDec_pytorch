#!/bin/bash

# just modify the path to the model

python src/translate.py \
  --trdec \
  --model_dir="outputs" \
  --data_path="data/orm_data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_vocab="vocab.rule.eng" \
  --source_test="set0-trainunfilt-head10.tok.piece.orm" \
  --target_test="set0-trainunfilt-head10.tok.piece.eng" \
  --target_tree_test="set0-trainunfilt-head10.tok.parse.eng" \
  --beam_size=4 \
  --max_len=200 \
  --n_train_sents=10000 \
  "$@"

