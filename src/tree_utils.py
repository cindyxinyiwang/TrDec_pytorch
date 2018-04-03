import numpy as np
from collections import defaultdict
import re


class Vocab(object):
  '''
  Converts between strings and integer ids.
  
  Configured via either i2w or vocab_file (mutually exclusive).
  
  Args:
    i2w (list of string): list of words, including <s> and </s>
    vocab_file (str): file containing one word per line, and not containing <s>, </s>, <unk>
  '''
  ES_STR = "</s>"
  def __init__(self, hparams, i2w=None, vocab_file=None, frozen=True):
    assert i2w is None or vocab_file is None
    self.hparams = hparams
    self.frozen = frozen
    if vocab_file:
      i2w = self.i2w_from_vocab_file(vocab_file)
    if (i2w is not None):
      self.i2w = i2w
      self.w2i = {word: word_id for (word_id, word) in enumerate(self.i2w)}
    else :
      self.w2i = {"<pad>": 0,"<unk>": 1,"<s>": 2,"</s>": 3}
      self.i2w = ["<pad>", "<unk>", "<s>", "</s>"]

  def i2w_from_vocab_file(self, vocab_file):
    """
    Args:
      vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    """
    vocab = []
    with open(vocab_file, encoding='utf-8') as f:
      for line in f:
        word = line.strip()
        vocab.append(word)
    return vocab

  def convert(self, w):
    if w not in self.w2i:
      if self.frozen:
       return self.hparams.unk_id
      self.w2i[w] = len(self.i2w)
      self.i2w.append(w)
    return self.w2i[w]

  def __getitem__(self, i):
    return self.i2w[i]

  def __len__(self):
    return len(self.i2w)

class RuleVocab(object):
  '''
  Converts between strings and integer ids
  '''
  def __init__(self, hparams, vocab_file=None, frozen=True, offset=0):
    """
    :param i2w: list of words, including <s> and </s>
    :param vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    i2w and vocab_file are mutually exclusive
    """
    self.hparams = hparams
    self.frozen = frozen
    self.offset = offset
    if vocab_file:
      self.i2w, self.w2i, self.lhs_to_index = self.from_vocab_file(vocab_file)
    else:
      self.w2i = {"<pad>": 0,"<unk>": 1,"<s>": 2,"</s>": 3}
      self.i2w = ["<pad>", "<unk>", "<s>", "</s>"]
      self.lhs_to_index = defaultdict(list)

  def from_vocab_file(self, vocab_file):
    """
    :param vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
    """
    i2w = []
    w2i = {}
    lhs_to_index = defaultdict(list)
    special_toks = set([self.hparams.bos, self.hparams.eos, self.hparams.pad, self.hparams.unk])
    with open(vocab_file, encoding='utf-8') as f:
      i = 0
      for line in f:
        word = line.strip()
        if word in special_toks:
          i2w.append(word)
          w2i[word] = i
        else:
          rule = Rule.from_str(word)
          i2w.append(rule)
          w2i[rule] = i
          lhs_to_index[rule.lhs].append(i)
        i += 1
    return i2w, w2i, lhs_to_index

  def convert(self, w):
    """ w is a Rule object """
    if w not in self.w2i:
      if self.frozen:
        return self.hparams.unk_id + self.offset
      self.w2i[w] = len(self.i2w)
      self.lhs_to_index[w.lhs].append(len(self.i2w))
      self.i2w.append(w)
    return self.w2i[w] + self.offset

  def rule_index_with_lhs(self, lhs):
    return self.lhs_to_index[lhs]

  def __getitem__(self, i):
    i -= self.offset
    return self.i2w[i]

  def __len__(self):
    return len(self.i2w)

class Rule(object):
  yaml_tag = "!Rule"

  def __init__(self, lhs, rhs=[], open_nonterms=[]):
    self.lhs = lhs
    self.rhs = rhs
    self.open_nonterms = open_nonterms
    self.serialize_params = {'lhs': self.lhs, 'rhs': self.rhs, 'open_nonterms': self.open_nonterms}

  def __str__(self):
    return (self.lhs + '|||' + ' '.join(self.rhs) + '|||' + ' '.join(self.open_nonterms))

  @staticmethod
  def from_str(line):
    segs = line.split('|||')
    assert len(segs) == 3
    lhs = segs[0]
    rhs = segs[1].split()
    open_nonterms = segs[2].split()
    return Rule(lhs, rhs, open_nonterms)

  def __hash__(self):
    #return hash(str(self) + " ".join(open_nonterms))
    if not hasattr(self, 'lhs'):
      return id(self)
    else:
      return hash(str(self))

  def __eq__(self, other):
    if not hasattr(other, 'lhs'):
      return False
    if not self.lhs == other.lhs:
      return False
    if not " ".join(self.rhs) == " ".join(other.rhs):
      return False
    if not " ".join(self.open_nonterms) == " ".join(other.open_nonterms):
      return False
    return True

class TreeNode(object):
  """A class that represents a tree node object """
  def __init__(self, string, children, timestep=-1, id=-1, last_word_t=0):
    self.label = string
    self.children = children
    for c in children:
      if hasattr(c, "set_parent"):
        c.set_parent(self)
    self._parent = None
    self.timestep = timestep
    self.id = id
    self.last_word_t = last_word_t
    self.frontir_label = None
  def is_preterminal(self):
    # return len(self.children) == 1 and (not hasattr(self.children[0], 'is_preterminal'))
    for c in self.children:
      if hasattr(c, 'is_preterminal'): return False
    return True
  def to_parse_string(self):
    c_str = []
    stack = [self]
    while stack:
      cur = stack.pop()
      while not hasattr(cur, 'label'):
        c_str.append(cur)
        if not stack: break
        cur = stack.pop()
      if not hasattr(cur, 'children'): break
      stack.append(u')')
      for c in reversed(cur.children):
        stack.append(c)
      stack.append(u'({} '.format(cur.label))
    return u"".join(c_str)
  def to_string(self, piece=True):
    """
    convert subtree into the sentence it represents
    """
    toks = []
    stack = [self]
    while stack:
      cur = stack.pop()
      while not hasattr(cur, 'label'):
        toks.append(cur)
        if not stack: break
        cur = stack.pop()
      if not hasattr(cur, 'children'): break
      for c in reversed(cur.children):
        stack.append(c)
    if not piece:
      return u" ".join(toks)
    else:
      return u"".join(toks).replace(u'\u2581', u' ').strip()
  def parent(self):
    return self._parent
  def set_parent(self, new_parent):
    self._parent = new_parent
  def add_child(self, child, id2n=None, last_word_t=0):
    self.children.append(child)
    if hasattr(child, "set_parent"):
      child._parent = self
      child.last_word_t = last_word_t
      if id2n:
        child.id = len(id2n)
        id2n[child.id] = child
        return child.id
    return -1
  def copy(self):
    new_node = TreeNode(self.label, [])
    for c in self.children:
      if hasattr(c, 'copy'):
        new_node.add_child(c.copy())
      else:
        new_node.add_child(c)
    return new_node
  def frontir_nodes(self):
    frontir = []
    for c in self.children:
      if hasattr(c, 'children'):
        if len(c.children) == 0:
          frontir.append(c)
        else:
          frontir.extend(c.frontir_nodes())
    return frontir
  def leaf_nodes(self):
    leaves = []
    for c in self.children:
      if hasattr(c, 'children'):
        leaves.extend(c.leaf_nodes())
      else:
        leaves.append(c)
    return leaves
  def get_leaf_lens(self, len_dict):
    if self.is_preterminal():
      l = self.leaf_nodes()
      # if len(l) > 10:
      #    print l, len(l)
      len_dict[len(l)] += 1
      return
    for c in self.children:
      if hasattr(c, 'is_preterminal'):
        c.get_leaf_lens(len_dict)
  def set_timestep(self, t, t2n=None, id2n=None, last_word_t=0, sib_t=0, open_stack=[]):
    """
    initialize timestep for each node
    """
    self.timestep = t
    self.last_word_t = last_word_t
    self.sib_t = sib_t
    next_word_t = last_word_t
    if not t2n is None:
      assert self.timestep == len(t2n)
      assert t not in t2n
      t2n[t] = self
    if not id2n is None:
      self.id = t
      id2n[t] = self
    sib_t = 0
    assert self.label == open_stack[-1]
    open_stack.pop()
    new_open_label = []
    for c in self.children:
      if hasattr(c, 'set_timestep'):
        new_open_label.append(c.label)
    new_open_label.reverse()
    open_stack.extend(new_open_label)
    if open_stack:
      self.frontir_label = open_stack[-1]
    else:
      self.frontir_label = Vocab.ES_STR
    c_t = t
    for c in self.children:
      # c_t = t + 1  # time of current child
      if hasattr(c, 'set_timestep'):
        c_t = t + 1
        t, next_word_t = c.set_timestep(c_t, t2n, id2n, next_word_t, sib_t, open_stack)
      else:
        next_word_t = t
      sib_t = c_t
    return t, next_word_t

class Tree(object):
  """A class that represents a parse tree"""
  yaml_tag = u"!Tree"
  def __init__(self, root=None, sent_piece=None, binarize=False):
    self.id2n = {}
    self.t2n = {}
    self.open_nonterm_ids = []
    self.last_word_t = -1
    if root:
      self.root = TreeNode('XXX', [root])
    else:
      self.last_word_t = 0
      self.root = TreeNode('XXX', [], id=0, timestep=0)
      self.id2n[0] = self.root
  def reset_timestep(self):
    self.root.set_timestep(0, self.t2n, self.id2n, open_stack=['XXX'])
  def __str__(self):
    return self.root.to_parse_string()
  def to_parse_string(self):
    return self.root.to_parse_string()
  def copy(self):
    '''Return a deep copy of the current tree'''
    copied_tree = Tree()
    copied_tree.id2n = {}
    copied_tree.t2n = {}
    copied_tree.open_nonterm_ids = self.open_nonterm_ids[:]
    copied_tree.last_word_t = self.last_word_t
    root = TreeNode('trash', [])
    stack = [self.root]
    copy_stack = [root]
    while stack:
      cur = stack.pop()
      copy_cur = copy_stack.pop()
      copy_cur.label = cur.label
      copy_cur.children = []
      copy_cur.id = cur.id
      copy_cur.timestep = cur.timestep
      copy_cur.last_word_t = cur.last_word_t
      copied_tree.id2n[copy_cur.id] = copy_cur
      if copy_cur.timestep >= 0:
        copied_tree.t2n[copy_cur.timestep] = copy_cur
      for c in cur.children:
        if hasattr(c, 'set_parent'):
          copy_c = TreeNode(c.label, [])
          copy_cur.add_child(copy_c)
          stack.append(c)
          copy_stack.append(copy_c)
        else:
          copy_cur.add_child(c)
    copied_tree.root = root
    return copied_tree
  @classmethod
  def from_rule_deriv(cls, derivs, wordswitch=True):
    tree = Tree()
    stack_tree = [tree.root]
    for x in derivs:
      r, stop = x
      p_tree = stack_tree.pop()
      if type(r) != Rule:
        if p_tree.label != '*':
          for i in derivs:
            if type(i[0]) != Rule:
              print
              i[0].encode('utf-8'), i[1]
            else:
              print
              i[0], i[1]
        assert p_tree.label == '*', p_tree.label
        if wordswitch:
          if r != Vocab.ES_STR:
            p_tree.add_child(r)
            stack_tree.append(p_tree)
        else:
          p_tree.add_child(r)
          if not stop:
            stack_tree.append(p_tree)
        continue
      if p_tree.label == 'XXX':
        new_tree = TreeNode(r.lhs, [])
        p_tree.add_child(new_tree)
      else:
        if p_tree.label != r.lhs:
          for i in derivs:
            if type(i[0]) != Rule:
              print
              i[0].encode('utf-8'), i[1]
            else:
              print
              i[0], i[1]
          print
          tree.to_parse_string().encode('utf-8')
          print
          p_tree.label.encode('utf-8'), r.lhs.encode('utf-8')
          exit(1)
        assert p_tree.label == r.lhs, "%s %s" % (p_tree.label, r.lhs)
        new_tree = p_tree
      open_nonterms = []
      for child in r.rhs:
        if child not in r.open_nonterms:
          new_tree.add_child(child)
        else:
          n = TreeNode(child, [])
          new_tree.add_child(n)
          open_nonterms.append(n)
      open_nonterms.reverse()
      stack_tree.extend(open_nonterms)
    return tree
  def to_string(self, piece=False):
    """
    convert subtree into the sentence it represents
    """
    return self.root.to_string(piece)
  def add_rule(self, id, rule):
    ''' Add one node to the tree based on current rule; only called on root tree '''
    node = self.id2n[id]
    node.set_timestep(len(self.t2n), self.t2n)
    node.last_word_t = self.last_word_t
    assert rule.lhs == node.label, "Rule lhs %s does not match the node %s to be expanded" % (rule.lhs, node.label)
    new_open_ids = []
    for rhs in rule.rhs:
      if rhs in rule.open_nonterms:
        new_open_ids.append(self.id2n[id].add_child(TreeNode(rhs, []), self.id2n))
      else:
        self.id2n[id].add_child(rhs)
        self.last_word_t = node.timestep
    new_open_ids.reverse()
    self.open_nonterm_ids.extend(new_open_ids)
    if self.open_nonterm_ids:
      node.frontir_label = self.id2n[self.open_nonterm_ids[-1]].label
    else:
      node.frontir_label = None
  def get_next_open_node(self):
    if len(self.open_nonterm_ids) == 0:
      print("stack empty, tree is complete")
      return -1
    return self.open_nonterm_ids.pop()
  def get_timestep_data(self, id):
    ''' Return a list of timesteps data associated with current tree node; only called on root tree '''
    data = []
    if self.id2n[id].parent():
      data.append(self.id2n[id].parent().timestep)
    else:
      data.append(0)
    data.append(self.id2n[id].last_word_t)
    return data
  def get_data_root(self, rule_vocab, word_vocab=None):
    data = []
    for t in range(1, len(self.t2n)):
      node = self.t2n[t]
      children, open_nonterms = [], []
      for c in node.children:
        if type(c) == str:
          children.append(c)
        else:
          children.append(c.label)
          open_nonterms.append(c.label)
      paren_t = 0 if not node.parent() else node.parent().timestep
      is_terminal = 1 if len(open_nonterms) == 0 else 0
      if word_vocab and is_terminal:
        for c in node.children:
          d = [word_vocab.convert(c), paren_t, is_terminal]
          data.append(d)
      else:
        r = Rule(node.label, children, open_nonterms)
        d = [rule_vocab.convert(Rule(node.label, children, open_nonterms)), paren_t, is_terminal]
        data.append(d)
    return data
  def get_bpe_rule(self, rule_vocab):
    ''' Get the rules for doing bpe. Label left and right child '''
    rule_idx = []
    for t in range(1, len(self.t2n)):
      node = self.t2n[t]
      children, open_nonterms = [], []
      child_idx = 1
      attach_tag = len(children) > 1
      for c in node.children:
        if type(c) == str:
          if attach_tag:
            children.append('{}_{}'.format(c, child_idx))
          else:
            children.append(c)
        else:
          if attach_tag:
            children.append('{}_{}'.format(c.label, child_idx))
          else:
            children.append(c.label)
          open_nonterms.append(c.label)
        child_idx += 1
      r = rule_vocab.convert(Rule(node.label, children, open_nonterms))
      rule_idx.append(r)
    return rule_idx
  def query_open_node_label(self):
    return self.id2n[self.open_nonterm_ids[-1]].label
def sent_piece_segs(p):
  '''
  Segment a sentence piece string into list of piece string for each word
  '''
  toks = re.compile(r'\u2581')
  ret = []
  p_start = 0
  for m in toks.finditer(p):
    pos = m.start()
    if pos == 0:
      continue
    ret.append(p[p_start:pos])
    p_start = pos
  if p_start != len(p) - 1:
    ret.append(p[p_start:])
  return ret

def sent_piece_segs_bpe(p):
  '''
Segment a sentence piece string into list of piece string for each word
'''
  # print p
  # print p.split()
  # toks = re.compile(ur'\xe2\x96\x81[^(\xe2\x96\x81)]+')
  toks = p.split()
  ret = []
  cur = []
  for t in toks:
    cur.append(t)
    if not t.endswith(u'@@'):
      ret.append(u' '.join(cur))
      cur = []
  return ret

def sent_piece_segs_post(p):
  '''
Segment a sentence piece string into list of piece string for each word
'''
  # print p
  # print p.split()
  # toks = re.compile(ur'\xe2\x96\x81[^(\xe2\x96\x81)]+')
  toks = re.compile(r'\u2581')
  ret = []
  p_start = 0
  for m in toks.finditer(p):
    pos = m.start()
    if pos == 0:
      continue
    ret.append(p[p_start:pos + 1].strip())
    p_start = pos + 1
  if p_start != len(p) - 1:
    ret.append(p[p_start:])
  return ret

def split_sent_piece(root, piece_l, word_idx):
  '''
  Split words into sentence piece
  '''
  new_children = []

  for i, c in enumerate(root.children):
    if type(c) == str:
      #print(root.to_parse_string())
      #print(piece_l, word_idx)
      piece = piece_l[word_idx].split()
      word_idx += 1
      new_children.extend(piece)
    else:
      word_idx = split_sent_piece(c, piece_l, word_idx)
      new_children.append(c)
  root.children = new_children
  return word_idx

def right_binarize(root, read_word=False):
  '''
  Right binarize a CusTree object
  read_word: if true, do not binarize terminal rules
  '''
  if type(root) == str:
    return root
  if read_word and root.label == u'*':
    return root
  if len(root.children) <= 2:
    new_children = []
    for c in root.children:
      new_children.append(right_binarize(c))
    root.children = new_children
  else:
    if "__" in root.label:
      new_label = root.label
    else:
      new_label = root.label + "__"
    n_left_child = TreeNode(new_label, root.children[1:])
    n_left_child._parent = root
    root.children = [right_binarize(root.children[0]), right_binarize(n_left_child)]
  return root

def add_preterminal(root):
  ''' Add preterminal X before each terminal symbol '''
  for i, c in enumerate(root.children):
    if type(c) == str:
      n = TreeNode(u'*', [c])
      n.set_parent(root)
      root.children[i] = n
    else:
      add_preterminal(c)

def add_preterminal_wordswitch(root, add_eos):
  ''' Add preterminal X before each terminal symbol '''
  ''' word_switch: one * symbol for each phrase chunk
      preterm_paren: * preterm parent already created
  '''
  preterm_paren = None
  new_children = []
  if root.label == u'*':
    if add_eos:
      root.add_child(Vocab.ES_STR)
    return root
  for i, c in enumerate(root.children):
    if type(c) == str:
      if not preterm_paren:
        preterm_paren = TreeNode('*', [])
        preterm_paren.set_parent(root)
        new_children.append(preterm_paren)
      preterm_paren.add_child(c)
    else:
      if preterm_paren and add_eos:
        preterm_paren.add_child(Vocab.ES_STR)
      c = add_preterminal_wordswitch(c, add_eos)
      new_children.append(c)
      preterm_paren = None
  if preterm_paren and add_eos:
    preterm_paren.add_child(Vocab.ES_STR)
  root.children = new_children
  return root

def remove_preterminal_POS(root):
  ''' Remove the POS tag before terminal '''
  for i, c in enumerate(root.children):
    if c.is_preterminal():
      root.children[i] = c.children[0]
    else:
      remove_preterminal_POS(c)
def replace_POS(root):
  ''' simply replace POS with * '''
  for i, c in enumerate(root.children):
    if c.is_preterminal():
      c.label = '*'
    else:
      replace_POS(c)
def merge_depth(root, max_depth, cur_depth):
  ''' raise up trees whose depth exceed max_depth '''
  if cur_depth >= max_depth:
    # root.label = u'*'
    root.children = root.leaf_nodes()
    return root
  new_children = []
  for i, c in enumerate(root.children):
    if hasattr(c, 'children'):
      c = merge_depth(c, max_depth, cur_depth + 1)
      # combine consecutive * nodes
      if new_children and hasattr(new_children[-1], 'label') and new_children[
        -1].is_preterminal() and c.is_preterminal():
        for x in c.children:
          new_children[-1].add_child(x)
      else:
        new_children.append(c)
    else:
      new_children.append(c)
  root.children = new_children
  return root
def merge_tags(root):
  ''' raise up trees whose label is in a given set '''
  kept_label = set([u'np', u'vp', u'pp', u's', u'root', u'sbar', u'sinv', u'XXX', u'prn', u'adjp', u'advp',
                    u'whnp', u'whadvp',
                    u'NP', u'VP', u'PP', u'S', u'ROOT', u'SBAR', u'FRAG', u'SINV', u'PRN'])
  if not root.label in kept_label:
    root.label = u'xx'
  for i, c in enumerate(root.children):
    if hasattr(c, 'children'):
      c = merge_tags(c)
    root.children[i] = c
  return root
def combine_tags(root):
  tag_dict = {'adjp': 'advp', 'sq': 'sbarq', 'whadjp': 'whadvp'}
# Tokenize a string.
# Tokens yielded are of the form (type, string)
# Possible values for 'type' are '(', ')' and 'WORD'
def tokenize(s):
  toks = re.compile(r' +|[^() ]+|[()]')
  for match in toks.finditer(s):
    s = match.group(0)
    if s[0] == ' ':
      continue
    if s[0] in '()':
      yield (s, s)
    else:
      yield ('WORD', s)
# Parse once we're inside an opening bracket.
def parse_inner(toks):
  ty, name = next(toks)
  if ty != 'WORD': raise ParseError
  children = []
  while True:
    ty, s = next(toks)
    # print ty, s
    if ty == '(':
      children.append(parse_inner(toks))
    elif ty == ')':
      return TreeNode(name, children)
    else:
      children.append(s)
class ParseError(Exception):
  pass
# Parse this grammar:
# ROOT ::= '(' INNER
# INNER ::= WORD ROOT* ')'
# WORD ::= [A-Za-z]+
def parse_root(toks):
  ty, s = next(toks)
  if ty != '(':
    # print ty, s
    raise ParseError
  return parse_inner(toks)
