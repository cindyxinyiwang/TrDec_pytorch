import sys
from tree_utils import *
from hparams import HParams

piece_file = "data/orm_data/set0-trainunfilt.tok.piece.eng"
tree_file = "data/orm_data/set0-trainunfilt.tok.eng.random_bina"
rule_vocab_file = "data/orm_data/vocab.random_bina_rule.eng"
word_vocab_file = "data/orm_data/vocab.random_bina_word.eng"
#
#piece_file = "data/kftt_data/kyoto-train.lowpiece.en"
#tree_file = "data/kftt_data/kyoto-train.lower.en.bina"
#rule_vocab_file = "data/kftt_data/vocab.bina_rule.en"
#word_vocab_file = "data/kftt_data/vocab.bina_word.en"


hp = HParams()
rule_vocab = RuleVocab(hparams=hp, frozen=False)
word_vocab = Vocab(hparams=hp, frozen=False)

piece_file = open(piece_file, 'r', encoding='utf-8')
tree_file = open(tree_file, 'r', encoding='utf-8')
for piece_line, tree_line in zip(piece_file, tree_file):
  tree = Tree(parse_root(tokenize(tree_line)))
  #remove_preterminal_POS(tree.root)
  #merge_depth(tree.root, 4, 0)
  pieces = sent_piece_segs(piece_line)
  split_sent_piece(tree.root, pieces, 0)
  add_preterminal_wordswitch(tree.root, add_eos=True)
  remove_lhs(tree.root, 'ROOT')
  tree.root.label = "XXX"
  tree.reset_timestep()
  tree.get_data_root(rule_vocab, word_vocab)

binarize = False
del_preterm_POS=True
replace_pos=False
read_word=True
merge=False
merge_level=-1
add_eos=True
bpe_post=True

with open(rule_vocab_file, 'w', encoding='utf-8') as myfile:
  for r in rule_vocab:
    myfile.write(str(r) + '\n')

if word_vocab_file:
  with open(word_vocab_file, 'w', encoding='utf-8') as myfile:
    for w in word_vocab:
      myfile.write(w + '\n')
