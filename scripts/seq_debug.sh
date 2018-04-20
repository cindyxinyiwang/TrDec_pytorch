#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16g
#SBATCH -t 0


module load singularity
singularity shell --nv /projects/tir1/singularity/ubuntu-16.04-lts_tensorflow-1.4.0_cudnn-8.0-v6.0.img


python src/main.py \
  --clean_mem_every=5 \
  --output_dir="outputs_kftt_seq_debug" \
  --log_every=100 \
  --eval_every=5000 \
  --reset_output_dir \
  --d_word_vec=512 \
  --d_model=512 \
  --data_path="data/kftt_data/" \
  --target_vocab="full-vocab.en" \
  --source_train="kyoto-lseq.lowpiece.ja" \
  --target_train="kyoto-lseq.lowpiece.en" \
  --source_valid="kyoto-lseq.lowpiece.ja" \
  --target_valid="kyoto-lseq.lowpiece.en" \
  --target_valid_ref="kyoto-lseq.lower.en" \
  --source_vocab="full-vocab.ja" \
  --source_test="kyoto-lseq.lowpiece.ja" \
  --target_test="kyoto-lseq.lowpiece.en" \
  --batch_size=16 \
  --n_train_sents=200000000 \
  --max_len=10000 \
  --max_tree_len=500 \
  --n_train_steps=80000 \
  --seed 0 \
  --cuda \
  "$@"
