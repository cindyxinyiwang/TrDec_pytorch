#!/bin/bash
export PYTHONPATH="$(pwd)"                                                       
export CUDA_VISIBLE_DEVICES="2"   

# The experiment that produce the best performing TrDec in the paper
# Use concatenation of two kinds of synthesized binary tree

python3.6 src/main.py \
  --trdec \
  --no_lhs \
  --pos=1 \
  --output_dir="outputs_orm_trdec_s0" \
  --log_every=20 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="data/orm_data/" \
  --target_tree_vocab="vocab.bina_rule.eng" \
  --target_word_vocab="vocab.bina_word.eng" \
  --target_tree_train="set0-trainunfilt.tok.eng.bina+w" \
  --target_tree_valid="set0-dev.tok.eng.bina" \
  --target_tree_test="set0-test.tok.eng.bina" \
  --source_train="set0-trainunfilt.tok.piece.orm.bina+w" \
  --target_train="set0-trainunfilt.tok.piece.eng.bina+w" \
  --source_valid="set0-dev.tok.piece.orm" \
  --target_valid="set0-dev.tok.piece.eng" \
  --source_vocab="vocab.orm" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --batch_size=800 \
  --batcher="word" \
  --n_train_sents=200000 \
  --max_len=2000 \
  --n_train_steps=5000 \
  --cuda \
  "$@"


