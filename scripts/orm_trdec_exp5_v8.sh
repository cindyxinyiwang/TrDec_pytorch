#!/bin/bash
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"

python3.6 src/main.py \
  --trdec \
  --no_lhs \
  --pos=1 \
  --rule_parent_feed=0 \
  --parent_feed=0 \
  --output_dir="outputs_orm_trdec_exp5_v8_s0" \
  --log_every=10 \
  --eval_every=200 \
  --load_model \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="data/orm_data/" \
  --target_tree_vocab="vocab.pos-no-lhs-rule.eng" \
  --target_word_vocab="vocab.pos-no-lhs-word.eng" \
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
  --batch_size=800 \
  --batcher="word" \
  --n_train_sents=200000 \
  --max_len=2000 \
  --n_train_steps=7000 \
  --seed 0 \
  --cuda \
  --lr=0.000125 \
  --loss_type="word" \
  --reset_hparams \
  "$@"


