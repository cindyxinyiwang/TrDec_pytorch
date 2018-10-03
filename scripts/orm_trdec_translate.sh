#!/bin/bash
export PYTHONPATH="$(pwd)"                                                       
export CUDA_VISIBLE_DEVICES="1" 


python3.6 src/translate.py \
  --trdec \
  --model_dir="outputs_orm_trdec_s0" \
  --data_path="data/orm_data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.bina_word.eng" \
  --target_tree_vocab="vocab.bina_rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.eng.bina" \
  --beam_size=45 \
  --max_len=500 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=1 \
  --out_file="beam45_m1" \
  --max_tree_len=3000 \
  --no_lhs \
  --pos=1 \
  --cuda \
  "$@"

