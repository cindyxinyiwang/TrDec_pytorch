#!/bin/bash

python src/main.py \
  --trdec \
  --output_dir="outputs" \
  --log_every=50 \
  --eval_every=500 \
  --reset_output_dir \
  --d_word_vec=4 \
  --d_model=4 \
  --data_path="data/orm_data/" \
  --target_tree_vocab="vocab.rule.eng" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_train="set0-trainunfilt.tok.parse.eng" \
  --target_tree_valid="set0-dev.tok.parse.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --source_train="set0-trainunfilt.tok.piece.orm" \
  --target_train="set0-trainunfilt.tok.piece.eng" \
  --source_valid="set0-dev.tok.piece.orm" \
  --target_valid="set0-dev.tok.piece.eng" \
  --source_vocab="vocab.orm" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --batch_size=2 \
  --n_train_sents=200000 \
  --max_len=200 \
  --n_train_steps=5000 \
  "$@"

