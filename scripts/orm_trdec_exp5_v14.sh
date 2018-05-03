#!/bin/bash
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES="2"


python3.6 src/main.py \
  --trdec \
  --no_lhs \
  --pos=1 \
  --parent_feed=0 \
  --rule_parent_feed=0 \
  --output_dir="outputs_orm_trdec_exp5_v14_s0" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="data/orm_data/" \
  --target_tree_vocab="vocab.random_bina_rule.eng" \
  --target_word_vocab="vocab.random_bina_word.eng" \
  --target_tree_train="set0-trainunfilt.tok.eng.random_bina" \
  --target_tree_valid="set0-dev.tok.eng.random_bina" \
  --target_tree_test="set0-test.tok.eng.random_bina" \
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
  "$@"


python3.6 src/main.py \
  --trdec \
  --no_lhs \
  --pos=1 \
  --parent_feed=0 \
  --rule_parent_feed=0 \
  --output_dir="outputs_orm_trdec_exp5_v14_s1" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="data/orm_data/" \
  --target_tree_vocab="vocab.random_bina_rule.eng" \
  --target_word_vocab="vocab.random_bina_word.eng" \
  --target_tree_train="set0-trainunfilt.tok.eng.random_bina" \
  --target_tree_valid="set0-dev.tok.eng.random_bina" \
  --target_tree_test="set0-test.tok.eng.random_bina" \
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
  --seed 1 \
  --cuda \
  "$@"


python3.6 src/main.py \
  --trdec \
  --no_lhs \
  --pos=1 \
  --parent_feed=0 \
  --rule_parent_feed=0 \
  --output_dir="outputs_orm_trdec_exp5_v14_s2" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="data/orm_data/" \
  --target_tree_vocab="vocab.random_bina_rule.eng" \
  --target_word_vocab="vocab.random_bina_word.eng" \
  --target_tree_train="set0-trainunfilt.tok.eng.random_bina" \
  --target_tree_valid="set0-dev.tok.eng.random_bina" \
  --target_tree_test="set0-test.tok.eng.random_bina" \
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
  --seed 2 \
  --cuda \
  "$@"


python3.6 src/main.py \
  --trdec \
  --no_lhs \
  --pos=1 \
  --parent_feed=0 \
  --rule_parent_feed=0 \
  --output_dir="outputs_orm_trdec_exp5_v14_s3" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="data/orm_data/" \
  --target_tree_vocab="vocab.random_bina_rule.eng" \
  --target_word_vocab="vocab.random_bina_word.eng" \
  --target_tree_train="set0-trainunfilt.tok.eng.random_bina" \
  --target_tree_valid="set0-dev.tok.eng.random_bina" \
  --target_tree_test="set0-test.tok.eng.random_bina" \
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
  --seed 3 \
  --cuda \
  "$@"


python3.6 src/main.py \
  --trdec \
  --no_lhs \
  --pos=1 \
  --parent_feed=0 \
  --rule_parent_feed=0 \
  --output_dir="outputs_orm_trdec_exp5_v14_s4" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="data/orm_data/" \
  --target_tree_vocab="vocab.random_bina_rule.eng" \
  --target_word_vocab="vocab.random_bina_word.eng" \
  --target_tree_train="set0-trainunfilt.tok.eng.random_bina" \
  --target_tree_valid="set0-dev.tok.eng.random_bina" \
  --target_tree_test="set0-test.tok.eng.random_bina" \
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
  --seed 4 \
  --cuda \
  "$@"


python3.6 src/main.py \
  --trdec \
  --no_lhs \
  --pos=1 \
  --parent_feed=0 \
  --rule_parent_feed=0 \
  --output_dir="outputs_orm_trdec_exp5_v14_s5" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="data/orm_data/" \
  --target_tree_vocab="vocab.random_bina_rule.eng" \
  --target_word_vocab="vocab.random_bina_word.eng" \
  --target_tree_train="set0-trainunfilt.tok.eng.random_bina" \
  --target_tree_valid="set0-dev.tok.eng.random_bina" \
  --target_tree_test="set0-test.tok.eng.random_bina" \
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
  --seed 5 \
  --cuda \
  "$@"


