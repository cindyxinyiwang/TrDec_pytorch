from tree_utils import *
import sys

def remove_pre(tree):
  if tree.is_preterminal():
    tree.label = '*'
    return
  for c in tree.children:
    if not type(c) == str:
      remove_pre(c)

for line in sys.stdin:
  tree = Tree(parse_root(tokenize(line))).root
  remove_pre(tree)
  print(tree.to_parse_string())
