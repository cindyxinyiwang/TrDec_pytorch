from tree_utils import *
import argparse
import os


def make_binary_tree(word_list, left, right):
  if left == right:
    return TreeNode("ROOT", word_list[left])
  root = TreeNode("ROOT", [])
  mid = int((left + right) / 2)
  left = make_binary_tree(word_list, left, mid)
  right = make_binary_tree(word_list, mid+1, right)
  root.children = [left, right]
  return root

parser = argparse.ArgumentParser(description="build trees")

parser.add_argument("--data_dir", type=str, help="directory of the data")
parser.add_argument("--file_name",type=str, help="name of the file to parse")
parser.add_argument("--tree_type",type=str, help="[phrase|right_branch]")
parser.add_argument("--parse_file_name",type=str, help="name of the file to parse")

tree_type = "bina"
data_dir="data/orm_data/"
input_files = ["set0-dev.tok.eng", "set0-test.tok.eng"]
output_files = ["set0-dev.tok.eng." + tree_type, "set0-test.tok.eng." + tree_type]

for in_file, out_file in zip(input_files, output_files):
  in_file = os.path.join(data_dir, in_file)
  out_file = os.path.join(data_dir, out_file)
  print("creating parse file {}".format(out_file))
  out_file = open(out_file, 'w')
  with open(in_file, encoding='utf-8') as myfile:
    for line in myfile:
      words = line.split()
      if tree_type == "right_branch":
        root = TreeNode("ROOT", [])
        c_n = root
        i = 0
        while i < len(words):
          c_n.children.append(TreeNode("*", words[i]))
          i += 1
          if i < len(words):
            n_n = TreeNode("ROOT", [])
            c_n.children.append(n_n)
            c_n = n_n
        out_file.write(root.to_parse_string() + '\n')
      elif tree_type == "bina":
        root = make_binary_tree(words, 0, len(words)-1)
        out_file.write(root.to_parse_string() + '\n')
      else:
        print("Not implemented")
  out_file.close()  
