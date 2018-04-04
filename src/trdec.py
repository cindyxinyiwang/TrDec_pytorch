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
      self.rule_ctx_to_readout = self.rule_lstm_cell.cuda()
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
    rule_hidden = dec_init 
    word_hidden = dec_init
    rule_input_feed = Variable(torch.zeros(batch_size, self.hparams.d_model * 2), requires_grad=False)
    word_input_feed = Variable(torch.zeros(batch_size, self.hparams.d_model * 2), requires_grad=False)
    #input_feed = Variable(dec_init[1][1].data.new(batch_size, self.hparams.d_model).zero_(), requires_grad=False)
    if self.hparams.cuda:
      rule_input_feed = rule_input_feed.cuda()
      word_input_feed = word_input_feed.cuda()
    # [batch_size, y_len, d_word_vec]
    trg_emb = self.emb(y_train[:, :, 0])
    logits = []
    states = [Variable(torch.zeros(batch_size, self.hparams.d_model), requires_grad=False)] # (timestep, batch_size, d_model)
    for t in range(y_max_len):
      y_emb_tm1 = trg_emb[:, t, :]

      all_state = torch.cat(states, dim=1).contiguous().view((t+1)*batch_size, self.hparams.d_model) # [batch_size*t, d_model]
      parent_t = y_train.data[:, t, 1] + (t+1) * torch.arange(batch_size).long() # [batch_size,]
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

  def step(self, x_enc, x_enc_k, y_tm1, dec_state, ctx_t):
    #y_emb_tm1 = self.word_emb(y_tm1)
    #y_input = torch.cat([y_emb_tm1, ctx_t], dim=1)
    #print (y_input.size())
    #print (dec_state[0].size())
    #print (dec_state[1].size())
    #h_t, c_t = self.layer(y_input, dec_state)
    #ctx = self.attention(h_t, x_enc_k, x_enc)
    #pre_readout = F.tanh(self.ctx_to_readout(torch.cat([h_t, ctx], dim=1)))
    #logits = self.readout(pre_readout)

    #return logits, (h_t, c_t), ctx
    pass 

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

  def translate(self, x_train, max_len=100, beam_size=5):
    hyps = []
    for x in x_train:
      x = Variable(torch.LongTensor(x), volatile=True)
      if self.hparams.cuda:
        x = x.cuda()
      hyp = self.translate_sent(x, max_len=max_len, beam_size=beam_size)[0]
      hyps.append(hyp.y[1:-1])
    return hyps

  def translate_sent(self, x_train, max_len=100, beam_size=5):
    assert len(x_train.size()) == 1
    x_len = [x_train.size(0)]
    x_train = x_train.unsqueeze(0)
    x_enc, dec_init = self.encoder(x_train, x_len)
    x_enc_k = self.enc_to_k(x_enc)
    length = 0
    completed_hyp = []
    input_feed = Variable(torch.zeros(1, self.hparams.d_model * 2), requires_grad=False)
    if self.hparams.cuda:
      input_feed = input_feed.cuda()
    active_hyp = [Hyp(state=dec_init, y=[self.hparams.bos_id], ctx_tm1=input_feed, score=0.)]
    while len(completed_hyp) < beam_size and length < max_len:
      length += 1
      new_hyp_score_list = []
      #hyp_num = len(active_hyp)
      #cur_x_enc = x_enc.repeat(hyp_num, 1, 1)
      #cur_x_enc_k = x_enc_k.repeat(hyp_num, 1, 1)
      #y_tm1 = Variable(torch.LongTensor([hyp.y[-1] for hyp in active_hyp]), volatile=True)
      #if self.hparams.cuda:
      #  y_tm1 = y_tm1.cuda()
      #logits = self.decoder.step(cur_x_enc, cur_x_enc_k, y_tm1, )
      for i, hyp in enumerate(active_hyp):
        y_tm1 = Variable(torch.LongTensor([int(hyp.y[-1])] ), volatile=True)
        if self.hparams.cuda:
          y_tm1 = y_tm1.cuda()
        logits, dec_state, ctx = self.decoder.step(x_enc, x_enc_k, y_tm1, hyp.state, hyp.ctx_tm1)
        hyp.state = dec_state
        hyp.ctx_tm1 = ctx 

        p_t = F.softmax(logits, -1).data
        new_hyp_scores = hyp.score + p_t 
        #print(new_hyp_scores)
        #print(p_t)
        new_hyp_score_list.append(new_hyp_scores)
        #print(hyp.y)
        #print(dec_state)
        #if len(active_hyp) > i+1:
        #  print(active_hyp[i+1].state)
      #print()
      #exit(0)
      live_hyp_num = beam_size - len(completed_hyp)
      new_hyp_scores = np.concatenate(new_hyp_score_list).flatten()
      new_hyp_pos = (-new_hyp_scores).argsort()[:live_hyp_num]
      prev_hyp_ids = new_hyp_pos / self.hparams.target_vocab_size
      word_ids = new_hyp_pos % self.hparams.target_vocab_size
      new_hyp_scores = new_hyp_scores[new_hyp_pos]

      new_hypotheses = []
      for prev_hyp_id, word_id, hyp_score in zip(prev_hyp_ids, word_ids, new_hyp_scores):
        prev_hyp = active_hyp[int(prev_hyp_id)]
        hyp = Hyp(state=prev_hyp.state, y=prev_hyp.y+[word_id], ctx_tm1=prev_hyp.ctx_tm1, score=hyp_score)
        if word_id == self.hparams.eos_id:
          completed_hyp.append(hyp)
        else:
          new_hypotheses.append(hyp)
        #print(word_id, hyp_score)
      #exit(0)
      active_hyp = new_hypotheses

    if len(completed_hyp) == 0:
      completed_hyp.append(active_hyp[0])
    return sorted(completed_hyp, key=lambda x: x.score, reverse=True)

class Hyp(object):
  def __init__(self, state, y, ctx_tm1, score):
    self.state = state
    self.y = y 
    self.ctx_tm1 = ctx_tm1
    self.score = score
