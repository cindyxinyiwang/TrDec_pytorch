#!/bin/bash

# just modify the path to the model

python src/translate.py \
  --model_dir="outputs_tiny" \
  --data_path="data/orm_data/" \
  --source_vocab="vocab.orm" \
  --target_vocab="vocab.eng" \
  --source_test="set0-trainunfilt-head10.tok.piece.orm" \
  --target_test="set0-trainunfilt-head10.tok.piece.eng" \
  --merge_bpe \
  --batch_size=32 \
  --beam_size=4 \
  --max_len=200 \
  --n_train_sents=10000 \
  "$@"

