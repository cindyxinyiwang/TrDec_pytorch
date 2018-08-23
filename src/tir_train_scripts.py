"""
make a bunch of training scripts on tir
"""
import os

script_dir = "scripts/"
base_output_dir = "uig_seq_exp2_v{}"
clean_mem_every=5
output_dir = ""
data_path = "data/uig_data/"
source_train = "all.ipa.spm8000.filt.all"
target_train = "all.mtok.spm8000.filt.eng"
source_valid = "set0-dev.ipa.allspm8000.uig"
target_valid = "set0-dev.mtok.spm8000.eng"
source_vocab = "vocab.all"
target_vocab = "vocab.eng"
max_len = 200
n_train_sents = 20000000
log_every=50
valid_batch_size = 7

eval_every= [1500]
d_word_vec = [128]
d_model = [512]
word_batch_size = [2000] 
sent_batch_size = [] 
seed = [0]
lr_dec_patience = [3]
dropout = [0.3]

template = "#!/bin/bash\n#SBATCH --gres=gpu:1\n#SBATCH --mem=16g\n#SBATCH -t 0\npython src/main.py \\\n"
template += "  --reset_output_dir \\\n"
template += "  --clean_mem_every={0} \\\n".format(clean_mem_every)
template += "  --data_path='{0}' \\\n".format(data_path)
template += "  --source_train='{0}' \\\n".format(source_train)
template += "  --target_train='{0}' \\\n".format(target_train)
template += "  --source_valid='{0}' \\\n".format(source_valid)
template += "  --target_valid='{0}' \\\n".format(target_valid)
template += "  --source_vocab='{0}' \\\n".format(source_vocab)
template += "  --target_vocab='{0}' \\\n".format(target_vocab)
template += "  --max_len={0} \\\n".format(max_len)
template += "  --n_train_sents={0} \\\n".format(n_train_sents)
template += "  --log_every={0} \\\n".format(log_every)
template += "  --valid_batch_size={0} \\\n".format(valid_batch_size)
template += "  --cuda \\\n"

i = 0
for d_m in d_model:
  for d_w in d_word_vec:
    for e in eval_every:
      for s in seed:
        for drop in dropout:
          for lr_dec_p in lr_dec_patience:
            script = template + "  --d_model={0} \\\n".format(d_m)
            script += "  --d_word_vec={0} \\\n".format(d_w)
            script += "  --eval_every={0} \\\n".format(e)
            script += "  --seed={0} \\\n".format(s)
            script += "  --dropout={0} \\\n".format(drop)
            script += "  --lr_dec_patience={0} \\\n".format(lr_dec_p)
            for batch_size in word_batch_size:
              output_dir = base_output_dir.format(i)
              script_name = os.path.join(script_dir, output_dir + ".sh")
              output_dir = "output_" + output_dir
              i += 1
              script += "  --batcher='word' \\\n"
              script += "  --batch_size={0} \\\n".format(batch_size)
              script += "  --output_dir='{0}' \\\n".format(output_dir)
            for batch_size in sent_batch_size:
              output_dir = base_output_dir.format(i)
              script_name = os.path.join(script_dir, output_dir + ".sh")
              output_dir = "output_" + output_dir
              i += 1
              script += "  --batcher='sent' \\\n"
              script += "  --batch_size={0} \\\n".format(batch_size)
              script += "  --output_dir='{0}' \\\n".format(output_dir)
            script += '  "$@"\n'
            with open(script_name, 'w') as myfile:
              myfile.write(script)
            print("writing to {}...".format(script_name))
