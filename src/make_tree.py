from tree_utils import *
import argparse
import os
import random
import re
import numpy as np
import math

def bina_list(node_list, left, right):
  if left == right:
    return node_list[left]
  root = TreeNode("ROOT", [])
  mid = int((left + right) / 2)
  left = bina_list(node_list, left, mid)
  right = bina_list(node_list, mid+1, right)
  root.children = [left, right]
  return root

def make_r_binary_tree(word_list, left, right):
  ## make if fully binary except for the end of tree
  def _bina_list(node_list, left, right):
    if left == right:
      return node_list[left]
    root = TreeNode("ROOT", [])
    mid = int((left + right) / 2)
    left = _bina_list(node_list, left, mid)
    right = _bina_list(node_list, mid+1, right)
    root.children = [left, right]
    return root
  l = len(word_list)
  num_preterm = int(pow(2, int(np.log2(l))) / 2)
  preterms = []
  for i in range(num_preterm-1):
    lc = TreeNode("ROOT", [word_list[i*2]])
    rc = TreeNode("ROOT", [word_list[i*2+1]])
    preterms.append(TreeNode("ROOT", [lc, rc]))
  preterms.append(make_binary_tree(word_list, (num_preterm-1)*2, right))
  return _bina_list(preterms, 0, len(preterms)-1)

def make_w_binary_tree(word_list):
  ## first combine words then make trees 
  l = len(word_list)
  nodes = []
  i = 0
  while i < l-1:
    lc = TreeNode("ROOT", [word_list[i]])
    rc = TreeNode("ROOT", [word_list[i+1]])
    nodes.append(TreeNode("ROOT", [lc, rc]))
    i += 2
  if l % 2 == 1:
    nodes.append(TreeNode("ROOT", [word_list[-1]]))
  return bina_list(nodes, 0, len(nodes)-1)

def make_right_binary_tree(word_list, left, right):
  if left == right:
    return TreeNode("ROOT", word_list[left])
  root = TreeNode("ROOT", [])
  mid = int((left + right) / 2)
  if mid == left: mid += 1
  left = make_right_binary_tree(word_list, left, mid-1)
  right = make_right_binary_tree(word_list, mid, right)
  root.children = [left, right]
  return root

def make_binary_tree(word_list, left, right):
  if left == right:
    return TreeNode("ROOT", word_list[left])
  root = TreeNode("ROOT", [])
  mid = int((left + right) / 2)
  left = make_binary_tree(word_list, left, mid)
  right = make_binary_tree(word_list, mid+1, right)
  root.children = [left, right]
  return root

def make_tri_tree(word_list, left, right):
  if left == right:
    return TreeNode("ROOT", word_list[left])
  if left == right-1:
    c1 = TreeNode("ROOT", word_list[left])
    c2 = TreeNode("ROOT", word_list[right])
    return TreeNode("ROOT", [c1, c2])
  root = TreeNode("ROOT", [])
  s1 = int((2*left + right) / 3)
  s2 = int((left + 2*right) / 3)
  c1 = make_tri_tree(word_list, left, s1)
  c2 = make_tri_tree(word_list, s1+1, s2)
  c3 = make_tri_tree(word_list, s2+1, right)
  root.children = [c1, c2, c3]
  return root
 
def make_random_binary_tree(word_list, left, right):
  if left == right:
    return TreeNode("ROOT", word_list[left])
  root = TreeNode("ROOT", [])
  mid = random.randint(left, right-1)
  left = make_binary_tree(word_list, left, mid)
  right = make_binary_tree(word_list, mid+1, right)
  root.children = [left, right]
  return root

def make_phrase_tree(string):
  def _bina_list(node_list, left, right):
    if left == right:
      return node_list[left]
    root = TreeNode("ROOT", [])
    mid = int((left + right) / 2)
    left = _bina_list(node_list, left, mid)
    right = _bina_list(node_list, mid+1, right)
    root.children = [left, right]
    return root

  #words = re.findall(r"[\w]+|[^\s\w]", string)
  #puncs =  re.findall(r"[^\s\w]", string)
  #print(string)
  split_points = []
  for match in re.finditer(r"[\s\w]+", string):
    if match.group().strip():
      #print(match.span(), match.group())
      split_points.append(match.span()[0])
  split_points = split_points[1:]
  nodes = []
  start = 0
  for s in split_points:
    cur_str = string[start:s].split()
    if string[s] != " ":
      if cur_str[-1] == "'":
        if string[s] == "s": 
          continue
        else:
          cur_str[-1] = cur_str[-1][:-1]
          s -= 1
      else:
        continue
    nodes.append(make_binary_tree(cur_str, 0, len(cur_str)-1))
    #print(string[start:s])
    start = s
  cur_str = string[start:].split()
  nodes.append(make_binary_tree(cur_str, 0, len(cur_str)-1))
  #print(string[start:])
  root = _bina_list(nodes, 0, len(nodes)-1)
  return root

parser = argparse.ArgumentParser(description="build trees")

parser.add_argument("--data_dir", type=str, help="directory of the data")
parser.add_argument("--file_name",type=str, help="name of the file to parse")
parser.add_argument("--tree_type",type=str, help="[phrase|random_bina|tri|bina|right_branch]")
parser.add_argument("--parse_file_name",type=str, help="name of the file to parse")

tree_type = "w_bina"
data_dir="data/kftt_data/"
input_files = ["kyoto-train.lower.en", "kyoto-dev.lower.en", "kyoto-test.lower.en"]
#data_dir="data/orm_data/"
#input_files = ["set0-trainunfilt.tok.eng", "set0-dev.tok.eng", "set0-test.tok.eng"]
#input_files = ["debug.tok.eng"]
output_files = []
for f in input_files:
  output_files.append(f + "." + tree_type)

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
      elif tree_type == "bina" :
        root = make_binary_tree(words, 0, len(words)-1)
        out_file.write(root.to_parse_string() + '\n')
      elif tree_type == "right_bina":
        root = make_right_binary_tree(words, 0, len(words)-1)
        out_file.write(root.to_parse_string() + '\n')
      elif tree_type == "tri":
        root = make_tri_tree(words, 0, len(words)-1)
        out_file.write(root.to_parse_string() + '\n')
      elif tree_type == "random_bina":
        root = make_random_binary_tree(words, 0, len(words)-1)
        out_file.write(root.to_parse_string() + '\n')
      elif tree_type == "phrase":
        root =  make_phrase_tree(line)
        out_file.write(root.to_parse_string() + '\n')
      elif tree_type == "r_bina":
        root =  make_r_binary_tree(words, 0, len(words)-1)
        out_file.write(root.to_parse_string() + '\n')
      elif tree_type == "w_bina":
        root =  make_w_binary_tree(words)
        out_file.write(root.to_parse_string() + '\n')
      else:
        print("Not implemented")
  out_file.close()  
