#!/bin/bash

python src/main.py \
  --output_dir="outputs_tiny" \
  --log_every=1 \
  --eval_every=5 \
  --reset_output_dir \
  --d_word_vec=288 \
  --d_model=288 \
  --data_path="data/orm_data/" \
  --source_train="set0-trainunfilt-head10.tok.piece.orm" \
  --target_train="set0-trainunfilt-head10.tok.piece.eng" \
  --source_valid="set0-trainunfilt-head10.tok.piece.orm" \
  --target_valid="set0-trainunfilt-head10.tok.piece.eng" \
  --source_vocab="vocab.orm" \
  --target_vocab="vocab.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --batch_size=2 \
  --n_train_sents=200000 \
  --max_len=20000 \
  --n_train_steps=5000 \
  --lr_dec=1 \
  "$@"

