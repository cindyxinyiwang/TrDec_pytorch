#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH -t 0


module load singularity
singularity shell --nv /projects/tir1/singularity/ubuntu-16.04-lts_tensorflow-1.4.0_cudnn-8.0-v6.0.img

python src/main.py \
  --trdec \
  --output_dir="outputs_orm_eng_trdec_v7_s0" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --target_tree_vocab="vocab.rule.eng" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_train="set0-trainunfilt.tok.parse.eng" \
  --target_tree_valid="set0-dev.tok.parse.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --source_train="set0-trainunfilt.tok.piece.orm" \
  --target_train="set0-trainunfilt.tok.piece.eng" \
  --source_valid="set0-dev.tok.piece.orm" \
  --target_valid="set0-dev.tok.piece.eng" \
  --source_vocab="vocab.word.orm" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --batch_size=16 \
  --n_train_sents=200000 \
  --max_len=2000 \
  --n_train_steps=5000 \
  --cuda \
  --seed 0 \
  "$@"

python src/main.py \
  --trdec \
  --output_dir="outputs_orm_eng_trdec_v7_s1" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --target_tree_vocab="vocab.rule.eng" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_train="set0-trainunfilt.tok.parse.eng" \
  --target_tree_valid="set0-dev.tok.parse.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --source_train="set0-trainunfilt.tok.piece.orm" \
  --target_train="set0-trainunfilt.tok.piece.eng" \
  --source_valid="set0-dev.tok.piece.orm" \
  --target_valid="set0-dev.tok.piece.eng" \
  --source_vocab="vocab.word.orm" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --batch_size=16 \
  --n_train_sents=200000 \
  --max_len=2000 \
  --n_train_steps=5000 \
  --cuda \
  --seed 1 \
  "$@"


python src/main.py \
  --trdec \
  --output_dir="outputs_orm_eng_trdec_v7_s2" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --target_tree_vocab="vocab.rule.eng" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_train="set0-trainunfilt.tok.parse.eng" \
  --target_tree_valid="set0-dev.tok.parse.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --source_train="set0-trainunfilt.tok.piece.orm" \
  --target_train="set0-trainunfilt.tok.piece.eng" \
  --source_valid="set0-dev.tok.piece.orm" \
  --target_valid="set0-dev.tok.piece.eng" \
  --source_vocab="vocab.word.orm" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --batch_size=16 \
  --n_train_sents=200000 \
  --max_len=2000 \
  --n_train_steps=5000 \
  --cuda \
  --seed 2 \
  "$@"


python src/main.py \
  --trdec \
  --output_dir="outputs_orm_eng_trdec_v7_s3" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --target_tree_vocab="vocab.rule.eng" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_train="set0-trainunfilt.tok.parse.eng" \
  --target_tree_valid="set0-dev.tok.parse.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --source_train="set0-trainunfilt.tok.piece.orm" \
  --target_train="set0-trainunfilt.tok.piece.eng" \
  --source_valid="set0-dev.tok.piece.orm" \
  --target_valid="set0-dev.tok.piece.eng" \
  --source_vocab="vocab.word.orm" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --batch_size=16 \
  --n_train_sents=200000 \
  --max_len=2000 \
  --n_train_steps=5000 \
  --cuda \
  --seed 3 \
  "$@"


python src/main.py \
  --trdec \
  --output_dir="outputs_orm_eng_trdec_v7_s4" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --target_tree_vocab="vocab.rule.eng" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_train="set0-trainunfilt.tok.parse.eng" \
  --target_tree_valid="set0-dev.tok.parse.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --source_train="set0-trainunfilt.tok.piece.orm" \
  --target_train="set0-trainunfilt.tok.piece.eng" \
  --source_valid="set0-dev.tok.piece.orm" \
  --target_valid="set0-dev.tok.piece.eng" \
  --source_vocab="vocab.word.orm" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --batch_size=16 \
  --n_train_sents=200000 \
  --max_len=2000 \
  --n_train_steps=5000 \
  --cuda \
  --seed 4 \
  "$@"


python src/main.py \
  --trdec \
  --output_dir="outputs_orm_eng_trdec_v7_s5" \
  --log_every=10 \
  --eval_every=200 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --target_tree_vocab="vocab.rule.eng" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_train="set0-trainunfilt.tok.parse.eng" \
  --target_tree_valid="set0-dev.tok.parse.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --source_train="set0-trainunfilt.tok.piece.orm" \
  --target_train="set0-trainunfilt.tok.piece.eng" \
  --source_valid="set0-dev.tok.piece.orm" \
  --target_valid="set0-dev.tok.piece.eng" \
  --source_vocab="vocab.word.orm" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --batch_size=16 \
  --n_train_sents=200000 \
  --max_len=2000 \
  --n_train_steps=5000 \
  --cuda \
  --seed 5 \
  "$@"


