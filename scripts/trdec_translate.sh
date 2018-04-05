#!/bin/bash

# just modify the path to the model
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="0"

python3.6 src/translate.py \
  --trdec \
  --model_dir="outputs_trdec" \
  --data_path="data/orm_data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_vocab="vocab.rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=4 \
  --max_len=200 \
  --n_train_sents=10000 \
  --cuda \
  "$@"

