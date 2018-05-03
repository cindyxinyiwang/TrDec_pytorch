#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=20g
#SBATCH -t 0


module load singularity
singularity shell --nv /projects/tir1/singularity/ubuntu-16.04-lts_tensorflow-1.4.0_cudnn-8.0-v6.0.img

python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_eng_trdec_s0" \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_vocab="vocab.rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=35 \
  --max_len=200 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=3 \
  --out_file="beam35_m3" \
  --cuda \
  "$@"

python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_eng_trdec_s1" \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_vocab="vocab.rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=35 \
  --max_len=200 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=3 \
  --out_file="beam35_m3" \
  --cuda \
  "$@"

python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_eng_trdec_s2" \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_vocab="vocab.rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=35 \
  --max_len=200 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=3 \
  --out_file="beam35_m3" \
  --cuda \
  "$@"

python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_eng_trdec_s3" \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_vocab="vocab.rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=35 \
  --max_len=200 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=3 \
  --out_file="beam35_m3" \
  --cuda \
  "$@"

python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_eng_trdec_s4" \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_vocab="vocab.rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=35 \
  --max_len=200 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=3 \
  --out_file="beam35_m3" \
  --cuda \
  "$@"

python src/translate.py \
  --trdec \
  --model_dir="outputs_orm_eng_trdec_s5" \
  --data_path="/projects/tir2/users/xinyiw1/loreili/orm-eng/data/" \
  --source_vocab="vocab.orm" \
  --target_word_vocab="vocab.word.eng" \
  --target_tree_vocab="vocab.rule.eng" \
  --source_test="set0-test.tok.piece.orm" \
  --target_test="set0-test.tok.piece.eng" \
  --target_tree_test="set0-test.tok.parse.eng" \
  --beam_size=35 \
  --max_len=200 \
  --n_train_sents=10000 \
  --merge_bpe \
  --poly_norm_m=3 \
  --out_file="beam35_m3" \
  --cuda \
  "$@"

