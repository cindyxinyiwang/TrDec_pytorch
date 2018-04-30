#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH -t 0


module load singularity
singularity shell --nv /projects/tir1/singularity/ubuntu-16.04-lts_tensorflow-1.4.0_cudnn-8.0-v6.0.img

python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_trdec_exp5_v7_s0" \
  --data_path="data/orm_data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.pos-no-lhs-word.eng" \
  --target_tree_vocab="vocab.pos-no-lhs-rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=25 \
  --max_len=500 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=1.5 \
  --out_file="beam25_m1.5" \
  --max_tree_len=3000 \
  --no_lhs \
  --pos=1 \
  --cuda \
  "$@"


python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_trdec_exp5_v7_s1" \
  --data_path="data/orm_data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.pos-no-lhs-word.eng" \
  --target_tree_vocab="vocab.pos-no-lhs-rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=25 \
  --max_len=500 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=1.5 \
  --out_file="beam25_m1.5" \
  --max_tree_len=3000 \
  --no_lhs \
  --pos=1 \
  --cuda \
  "$@"


python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_trdec_exp5_v7_s2" \
  --data_path="data/orm_data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.pos-no-lhs-word.eng" \
  --target_tree_vocab="vocab.pos-no-lhs-rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=25 \
  --max_len=500 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=1.5 \
  --out_file="beam25_m1.5" \
  --max_tree_len=3000 \
  --no_lhs \
  --pos=1 \
  --cuda \
  "$@"


python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_trdec_exp5_v7_s3" \
  --data_path="data/orm_data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.pos-no-lhs-word.eng" \
  --target_tree_vocab="vocab.pos-no-lhs-rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=25 \
  --max_len=500 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=1.5 \
  --out_file="beam25_m1.5" \
  --max_tree_len=3000 \
  --no_lhs \
  --pos=1 \
  --cuda \
  "$@"


python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_trdec_exp5_v7_s4" \
  --data_path="data/orm_data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.pos-no-lhs-word.eng" \
  --target_tree_vocab="vocab.pos-no-lhs-rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=25 \
  --max_len=500 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=1.5 \
  --out_file="beam25_m1.5" \
  --max_tree_len=3000 \
  --no_lhs \
  --pos=1 \
  --cuda \
  "$@"


python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_trdec_exp5_v7_s5" \
  --data_path="data/orm_data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.pos-no-lhs-word.eng" \
  --target_tree_vocab="vocab.pos-no-lhs-rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=25 \
  --max_len=500 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=1.5 \
  --out_file="beam25_m1.5" \
  --max_tree_len=3000 \
  --no_lhs \
  --pos=1 \
  --cuda \
  "$@"


