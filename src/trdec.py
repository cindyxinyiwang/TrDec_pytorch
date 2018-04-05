import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import numpy as np
from utils import *
from models import *

class TreeDecoder(nn.Module):
  def __init__(self, hparams):
    super(TreeDecoder, self).__init__()
    self.hparams = hparams
    self.target_vocab_size = self.hparams.target_word_vocab_size+self.hparams.target_rule_vocab_size
    self.emb = nn.Embedding(self.target_vocab_size,
                            self.hparams.d_word_vec,
                            padding_idx=hparams.pad_id)
    #self.attention = DotProdAttn(hparams)
    self.rule_attention = MlpAttn(hparams)
    self.word_attention = MlpAttn(hparams)
    # transform [word_ctx, word_h_t, rule_ctx, rule_h_t] to readout state vectors before softmax
    self.rule_ctx_to_readout = nn.Linear(hparams.d_model * 6, hparams.d_model, bias=False)
   
    self.word_ctx_to_readout = nn.Linear(hparams.d_model * 6, hparams.d_model, bias=False)

    self.readout = nn.Linear(hparams.d_model, 
                                  self.target_vocab_size, 
                                  bias=False)  
    # input: [y_t-1, parent_state, input_feed, word_state]
    self.rule_lstm_cell = nn.LSTMCell(hparams.d_word_vec + hparams.d_model * 4, 
                             hparams.d_model)
    # input: [y_t-1, parent_state, input_feed]
    self.word_lstm_cell = nn.LSTMCell(hparams.d_word_vec + hparams.d_model * 3, 
                             hparams.d_model)
    self.dropout = nn.Dropout(hparams.dropout)
    if self.hparams.cuda:
      self.emb = self.emb.cuda()
      self.rule_attention = self.rule_attention.cuda()
      self.word_attention = self.word_attention.cuda()
      self.rule_ctx_to_readout = self.rule_ctx_to_readout.cuda()
      self.word_ctx_to_readout = self.word_ctx_to_readout.cuda()
      self.readout = self.readout.cuda()
      self.rule_lstm_cell = self.rule_lstm_cell.cuda()
      self.word_lstm_cell = self.word_lstm_cell.cuda()
      self.dropout = self.dropout.cuda()

  def forward(self, x_enc, x_enc_k, dec_init, x_mask, y_train, y_mask):
    # get decoder init state and cell, use x_ct
    """
    x_enc: [batch_size, max_x_len, d_model * 2]

    """
    batch_size_x = x_enc.size()[0]
    batch_size, y_max_len, data_len = y_train.size()
    assert batch_size_x == batch_size
    #print(y_train)
    input_feed_zeros = torch.zeros(batch_size, self.hparams.d_model * 2)
    state_zeros = torch.zeros(batch_size, self.hparams.d_model)
    if self.hparams.cuda:
      input_feed_zeros = input_feed_zeros.cuda()
      state_zeros = state_zeros.cuda()

    rule_hidden = dec_init 
    word_hidden = dec_init
    rule_input_feed = Variable(input_feed_zeros, requires_grad=False)
    word_input_feed = Variable(input_feed_zeros, requires_grad=False)
    #input_feed = Variable(dec_init[1][1].data.new(batch_size, self.hparams.d_model).zero_(), requires_grad=False)
    if self.hparams.cuda:
      rule_input_feed = rule_input_feed.cuda()
      word_input_feed = word_input_feed.cuda()
    # [batch_size, y_len, d_word_vec]
    trg_emb = self.emb(y_train[:, :, 0])
    logits = []
    states = [Variable(state_zeros, requires_grad=False)] # (timestep, batch_size, d_model)
    offset = torch.arange(batch_size).long() 
    if self.hparams.cuda:
      offset = offset.cuda()
    for t in range(y_max_len):
      y_emb_tm1 = trg_emb[:, t, :]

      all_state = torch.cat(states, dim=1).contiguous().view((t+1)*batch_size, self.hparams.d_model) # [batch_size*t, d_model]
      if self.hparams.cuda:
        all_state = all_state.cuda()
      parent_t = y_train.data[:, t, 1] + (t+1) * offset # [batch_size,]
      parent_t = Variable(parent_t, requires_grad=False)
      #print(parent_t)
      parent_state = torch.index_select(all_state, dim=0, index=parent_t) # [batch_size, d_model]
      
      word_mask = y_train[:, t, 2].unsqueeze(1).expand(-1, self.hparams.d_model).float() # (1 is word, 0 is rule)
      word_mask_vocab = y_train[:, t, 2].unsqueeze(1).expand(-1, self.target_vocab_size).float()

      word_input = torch.cat([y_emb_tm1, parent_state, word_input_feed], dim=1)
      word_h_t, word_c_t = self.word_lstm_cell(word_input, word_hidden)
      #print(word_h_t.size())
      #print(word_mask.size())
      #print(word_hidden[0].size())
      word_h_t = word_h_t * word_mask + word_hidden[0] * (1-word_mask)
      word_c_t = word_c_t * word_mask + word_hidden[1] * (1-word_mask)

      rule_input = torch.cat([y_emb_tm1, parent_state, rule_input_feed, word_h_t], dim=1)
      rule_h_t, rule_c_t = self.rule_lstm_cell(rule_input, rule_hidden)

      rule_ctx = self.rule_attention(rule_h_t, x_enc_k, x_enc, attn_mask=x_mask)
      word_ctx = self.word_attention(word_h_t, x_enc_k, x_enc, attn_mask=x_mask)

      #print(rule_h_t.size())
      #print(rule_ctx.size())
      #print(word_h_t.size())
      #print(word_ctx.size())
      inp = torch.cat([rule_h_t, rule_ctx, word_h_t, word_ctx], dim=1)
      rule_pre_readout = F.tanh(self.rule_ctx_to_readout(inp))
      word_pre_readout = F.tanh(self.word_ctx_to_readout(inp))
      
      rule_pre_readout = self.dropout(rule_pre_readout)
      word_pre_readout = self.dropout(word_pre_readout)

      rule_score_t = self.readout(rule_pre_readout)
      word_score_t = self.readout(word_pre_readout)

      score_t = word_score_t*word_mask_vocab + rule_score_t*(1 - word_mask_vocab)
      logits.append(score_t)

      rule_input_feed = rule_ctx
      word_input_feed = word_ctx

      rule_hidden = (rule_h_t, rule_c_t)
      word_hidden = (word_h_t, word_c_t)

      states.append(rule_h_t)
    # [len_y, batch_size, trg_vocab_size]
    logits = torch.stack(logits).transpose(0, 1).contiguous()
    return logits

  def step(self, x_enc, x_enc_k, hyp, target_rule_vocab):
    y_tm1 = torch.LongTensor([int(hyp.y[-1])])
    if self.hparams.cuda:
      y_tm1 = y_tm1.cuda()
    y_tm1 = Variable(y_tm1, volatile=True)
    open_nonterms = hyp.open_nonterms
    rule_input_feed = hyp.rule_ctx_tm1
    word_input_feed = hyp.word_ctx_tm1
    rule_hidden = hyp.rule_hidden
    word_hidden = hyp.word_hidden
    word_h_t, word_c_t = word_hidden
    
    y_emb_tm1 = self.emb(y_tm1)
    cur_nonterm = open_nonterms[-1]
    parent_state = cur_nonterm.parent_state
    if hyp.y[-1] < self.hparams.target_word_vocab_size:
      # word
      word_input = torch.cat([y_emb_tm1, parent_state, word_input_feed], dim=1)
      word_h_t, word_c_t = self.word_lstm_cell(word_input, word_hidden)
      word_ctx = self.word_attention(word_h_t, x_enc_k, x_enc)
    else:
      word_ctx = hyp.word_ctx_tm1
    rule_input = torch.cat([y_emb_tm1, parent_state, rule_input_feed, word_h_t], dim=1)
    rule_h_t, rule_c_t = self.rule_lstm_cell(rule_input, rule_hidden)
    rule_ctx = self.rule_attention(rule_h_t, x_enc_k, x_enc)

    inp = torch.cat([rule_h_t, rule_ctx, word_h_t, word_ctx], dim=1)
    mask = torch.ones(1, self.target_vocab_size).byte()
    if cur_nonterm.label == '*':
      word_index = torch.arange(self.hparams.target_word_vocab_size).long()
      mask.index_fill_(1, word_index, 0)
      word_pre_readout = F.tanh(self.word_ctx_to_readout(inp))
      word_pre_readout = self.dropout(word_pre_readout)
      score_t = self.readout(word_pre_readout)
    else:
      rule_with_lhs = target_rule_vocab.rule_index_with_lhs(cur_nonterm.label)
      rule_index = torch.LongTensor(rule_with_lhs) + self.hparams.target_word_vocab_size
      #print(rule_index)
      #print(mask)
      mask.index_fill_(1, rule_index, 0)
      rule_pre_readout = F.tanh(self.rule_ctx_to_readout(inp))  
      rule_pre_readout = self.dropout(rule_pre_readout)
      score_t = self.readout(rule_pre_readout)
    if self.hparams.cuda:
      mask = mask.cuda()
    #print(score_t)
    #print(mask)
    score_t.data.masked_fill_(mask, float("-inf"))
    rule_hidden = (rule_h_t, rule_c_t)
    word_hidden = (word_h_t, word_c_t)
    return score_t, rule_hidden, word_hidden, rule_ctx, word_ctx, open_nonterms

class TrDec(nn.Module):
  
  def __init__(self, hparams):
    super(TrDec, self).__init__()
    self.encoder = Encoder(hparams)
    self.decoder = TreeDecoder(hparams)
    # transform encoder state vectors into attention key vector
    self.enc_to_k = nn.Linear(hparams.d_model * 2, hparams.d_model, bias=False)
    self.hparams = hparams
    if self.hparams.cuda:
      self.enc_to_k = self.enc_to_k.cuda()

  def forward(self, x_train, x_mask, x_len, y_train, y_mask, y_len):
    # [batch_size, x_len, d_model * 2]
    #print("x_train", x_train)
    #print("x_mask", x_mask)
    #print("x_len", x_len)
    x_enc, dec_init = self.encoder(x_train, x_len)
    x_enc_k = self.enc_to_k(x_enc)
    # [batch_size, y_len-1, trg_vocab_size]
    logits = self.decoder(x_enc, x_enc_k, dec_init, x_mask, y_train, y_mask)
    return logits

  def translate(self, x_train, target_rule_vocab, max_len=100, beam_size=5):
    hyps = []
    for x in x_train:
      #print(x)
      x = Variable(torch.LongTensor(x), volatile=True)
      if self.hparams.cuda:
        x = x.cuda()
      hyp = self.translate_sent(x, target_rule_vocab, max_len=max_len, beam_size=beam_size)[0]
      hyps.append(hyp.y[1:])
      #print(hyp.y)
    return hyps

  def translate_sent(self, x_train, target_rule_vocab, max_len=100, beam_size=5):
    assert len(x_train.size()) == 1
    x_len = [x_train.size(0)]
    x_train = x_train.unsqueeze(0)
    x_enc, dec_init = self.encoder(x_train, x_len)
    x_enc_k = self.enc_to_k(x_enc)
    length = 0
    completed_hyp = []
    input_feed_zeros = torch.zeros(1, self.hparams.d_model * 2)
    state_zeros = torch.zeros(1, self.hparams.d_model)
    if self.hparams.cuda:
      input_feed_zeros = input_feed_zeros.cuda()
      state_zeros = state_zeros.cuda()
    rule_input_feed = Variable(input_feed_zeros, requires_grad=False)
    word_input_feed = Variable(input_feed_zeros, requires_grad=False)
    active_hyp = [TrHyp(rule_hidden=dec_init,
                  word_hidden=dec_init,
                  y=[self.hparams.bos_id], 
                  rule_ctx_tm1=rule_input_feed, 
                  word_ctx_tm1=word_input_feed,
                  open_nonterms=[OpenNonterm(label='ROOT', 
                    parent_state=Variable(state_zeros, requires_grad=False))],
                  score=0.)]
    while len(completed_hyp) < beam_size and length < max_len:
      length += 1
      new_hyp_score_list = []
      for i, hyp in enumerate(active_hyp):
        logits, rule_hidden, word_hidden, rule_ctx, word_ctx, open_nonterms = self.decoder.step(x_enc, 
          x_enc_k, hyp, target_rule_vocab)
        hyp.rule_hidden = rule_hidden
        hyp.word_hidden = word_hidden
        hyp.rule_ctx_tm1 = rule_ctx 
        hyp.word_ctx_tm1 = word_ctx
        hyp.open_nonterms = open_nonterms

        p_t = F.softmax(logits, -1).data
        new_hyp_scores = hyp.score + p_t 
        new_hyp_score_list.append(new_hyp_scores)

      live_hyp_num = beam_size - len(completed_hyp)
      new_hyp_scores = np.concatenate(new_hyp_score_list).flatten()
      new_hyp_pos = (-new_hyp_scores).argsort()[:live_hyp_num]
      prev_hyp_ids = new_hyp_pos / self.decoder.target_vocab_size
      word_ids = new_hyp_pos % self.decoder.target_vocab_size
      new_hyp_scores = new_hyp_scores[new_hyp_pos]

      new_hypotheses = []
      for prev_hyp_id, word_id, hyp_score in zip(prev_hyp_ids, word_ids, new_hyp_scores):
        prev_hyp = active_hyp[int(prev_hyp_id)]
        open_nonterms = prev_hyp.open_nonterms[:]
        if word_id >= self.hparams.target_word_vocab_size:
          rule = target_rule_vocab[word_id]
          open_nonterms.pop()
          for c in reversed(rule.rhs):
            open_nonterms.append(OpenNonterm(label=c, parent_state=prev_hyp.rule_hidden[0]))
        else:
          assert open_nonterms[-1].label == '*'
          if word_id == self.hparams.eos_id: 
            open_nonterms.pop()

        hyp = TrHyp(rule_hidden=prev_hyp.rule_hidden, 
                    word_hidden=prev_hyp.word_hidden,
                    y=prev_hyp.y+[word_id], 
                    rule_ctx_tm1=prev_hyp.rule_ctx_tm1, 
                    word_ctx_tm1=prev_hyp.word_ctx_tm1,
                    open_nonterms=open_nonterms,
                    score=hyp_score)
        if len(hyp.open_nonterms) == 0:
          completed_hyp.append(hyp)
        else:
          new_hypotheses.append(hyp)
      active_hyp = new_hypotheses

    if len(completed_hyp) == 0:
      completed_hyp.append(active_hyp[0])
    return sorted(completed_hyp, key=lambda x: x.score, reverse=True)

class TrHyp(object):
  def __init__(self, rule_hidden, word_hidden, y, rule_ctx_tm1, word_ctx_tm1, score, open_nonterms):
    self.rule_hidden = rule_hidden
    self.word_hidden = word_hidden
    # [length_y, 2], each element (index, is_word)
    self.y = y 
    self.rule_ctx_tm1 = rule_ctx_tm1
    self.word_ctx_tm1 = word_ctx_tm1
    self.score = score
    self.open_nonterms = open_nonterms

class OpenNonterm(object):
  def __init__(self, label, parent_state):
    self.label = label
    self.parent_state = parent_state
