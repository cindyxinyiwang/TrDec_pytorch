import torch
import torch.nn.init as init
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import numpy as np
from utils import *
from models import *
import gc

class TreeDecoder(nn.Module):
  def __init__(self, hparams):
    super(TreeDecoder, self).__init__()
    self.hparams = hparams
    self.target_vocab_size = self.hparams.target_word_vocab_size+self.hparams.target_rule_vocab_size
    self.emb = nn.Embedding(self.target_vocab_size,
                            self.hparams.d_word_vec,
                            padding_idx=hparams.pad_id)
    if hasattr(self.hparams, "single_attn") and self.hparams.single_attn:
      if self.hparams.attn == "mlp":
        self.attention = MlpAttn(hparams)
      else:
        self.attention = DotProdAttn(hparams)
    else:
      if self.hparams.attn == "mlp":
        self.rule_attention = MlpAttn(hparams)
        self.word_attention = MlpAttn(hparams)
      else:
        self.rule_attention = DotProdAttn(hparams)
        self.word_attention = DotProdAttn(hparams)
    # transform [word_ctx, word_h_t, rule_ctx, rule_h_t] to readout state vectors before softmax
    if hasattr(self.hparams, "single_inp_readout") and self.hparams.single_inp_readout:
      inp_dim = hparams.d_model * 3
    else:
      inp_dim = hparams.d_model * 6
    if hasattr(self.hparams, "single_readout") and self.hparams.single_readout:
      self.ctx_to_readout = nn.Linear(inp_dim, hparams.d_model, bias=False)
    else:
      self.rule_ctx_to_readout = nn.Linear(inp_dim, hparams.d_model, bias=False)
      self.word_ctx_to_readout = nn.Linear(hparams.d_model * 6, hparams.d_model, bias=False)

    self.readout = nn.Linear(hparams.d_model, 
                                  self.target_vocab_size, 
                                  bias=False)  
    if self.hparams.share_emb_softmax:
      self.emb.weight = self.readout.weight
    # input: [y_t-1, parent_state, input_feed, word_state]
    rule_inp = hparams.d_word_vec + hparams.d_model * 3
    word_inp = hparams.d_word_vec + hparams.d_model * 2
    #if hparams.parent_feed:
    word_inp += hparams.d_model
    if hparams.rule_parent_feed:
      rule_inp += hparams.d_model
    self.rule_lstm_cell = nn.LSTMCell(rule_inp, hparams.d_model)
    # input: [y_t-1, parent_state, input_feed]
    self.word_lstm_cell = nn.LSTMCell(word_inp, hparams.d_model)
    self.dropout = nn.Dropout(hparams.dropout)

    vocab_mask = torch.zeros(1, 1, self.target_vocab_size)
    self.word_vocab_mask = vocab_mask.index_fill_(2, torch.arange(self.hparams.target_word_vocab_size).long(), 1)
    self.rule_vocab_mask = 1 - self.word_vocab_mask
    if self.hparams.cuda:
      self.rule_vocab_mask = self.rule_vocab_mask.cuda()
      self.word_vocab_mask = self.word_vocab_mask.cuda()
      self.emb = self.emb.cuda()
      if hasattr(self.hparams, "single_attn") and self.hparams.single_attn:
        self.attention = self.attention.cuda()
      else:
        self.rule_attention = self.rule_attention.cuda()
        self.word_attention = self.word_attention.cuda()
      if hasattr(self.hparams, "single_readout") and self.hparams.single_readout:
        self.ctx_to_readout = self.ctx_to_readout.cuda()
      else:
        self.rule_ctx_to_readout = self.rule_ctx_to_readout.cuda()
        self.word_ctx_to_readout = self.word_ctx_to_readout.cuda()
      self.readout = self.readout.cuda()
      self.rule_lstm_cell = self.rule_lstm_cell.cuda()
      self.word_lstm_cell = self.word_lstm_cell.cuda()
      self.dropout = self.dropout.cuda()

  def forward(self, x_enc, x_enc_k, dec_init, x_mask, y_train, y_mask, score_mask, y_label=None):
    # get decoder init state and cell, use x_ct
    """
    x_enc: [batch_size, max_x_len, d_model * 2]
    """
    batch_size_x = x_enc.size()[0]
    batch_size, y_max_len, data_len = y_train.size()
    assert batch_size_x == batch_size
    #print(y_train)
    input_feed_zeros = torch.zeros(batch_size, self.hparams.d_model * 2)
    if self.hparams.cuda:
      input_feed_zeros = input_feed_zeros.cuda()
    rule_hidden = dec_init 
    word_hidden = dec_init
    rule_input_feed = Variable(input_feed_zeros, requires_grad=False)
    word_input_feed = Variable(input_feed_zeros, requires_grad=False)
    if self.hparams.cuda:
      rule_input_feed = rule_input_feed.cuda()
      word_input_feed = word_input_feed.cuda()
    # [batch_size, y_len, d_word_vec]
    trg_emb = self.emb(y_train[:, :, 0])
    logits = []
    if hasattr(self.hparams, "single_readout") and self.hparams.single_readout:
      pre_readouts = []
    else:
      rule_pre_readouts = []
      word_pre_readouts = []
    if self.hparams.parent_feed:
      all_state = Variable(torch.zeros(batch_size, self.hparams.d_model),  requires_grad=False)
    offset = torch.arange(batch_size).long() 
    if self.hparams.cuda:
      offset = offset.cuda()
      if self.hparams.parent_feed:
        all_state = all_state.cuda()
    for t in range(y_max_len):
      y_emb_tm1 = trg_emb[:, t, :]
      
      if self.hparams.parent_feed:
        state_idx_t = t 
        state_idx_t += 1
        parent_t = y_train.data[:, t, 1] + state_idx_t * offset # [batch_size,]
        parent_t = Variable(parent_t, requires_grad=False)
        parent_state = torch.index_select(all_state.view(state_idx_t*batch_size, self.hparams.d_model), dim=0, index=parent_t) # [batch_size, d_model]
      
      word_mask = y_train[:, t, 2].unsqueeze(1).float() # (1 is word, 0 is rule)
      
      if self.hparams.parent_feed:
        word_input = torch.cat([y_emb_tm1, parent_state, word_input_feed], dim=1)
      else:
        word_input = torch.cat([y_emb_tm1, rule_hidden[0], word_input_feed], dim=1)
      word_h_t, word_c_t = self.word_lstm_cell(word_input, word_hidden)

      word_h_t = word_h_t * word_mask + word_hidden[0] * (1-word_mask)
      word_c_t = word_c_t * word_mask + word_hidden[1] * (1-word_mask)

      if self.hparams.rule_parent_feed:
        rule_input = torch.cat([y_emb_tm1, parent_state, rule_input_feed, word_h_t], dim=1)
      else:
        rule_input = torch.cat([y_emb_tm1, rule_input_feed, word_h_t], dim=1)
      rule_h_t, rule_c_t = self.rule_lstm_cell(rule_input, rule_hidden)

      if hasattr(self.hparams, "no_word_to_rule") and self.hparams.no_word_to_rule:
        eos_mask = (y_train[:, t, 0] == self.hparams.eos_id).unsqueeze(1).float()
        word_mask = word_mask - eos_mask
        rule_h_t = rule_h_t * (1-word_mask) + rule_hidden[0] * word_mask
        rule_c_t = rule_c_t * (1-word_mask) + rule_hidden[1] * word_mask

      if hasattr(self.hparams, "single_attn") and self.hparams.single_attn:
        rule_ctx = self.attention(rule_h_t, x_enc_k, x_enc, attn_mask=x_mask)
        word_ctx = self.attention(word_h_t, x_enc_k, x_enc, attn_mask=x_mask)
      else:
        rule_ctx = self.rule_attention(rule_h_t, x_enc_k, x_enc, attn_mask=x_mask)
        word_ctx = self.word_attention(word_h_t, x_enc_k, x_enc, attn_mask=x_mask)

      if hasattr(self.hparams, "single_inp_readout") and self.hparams.single_inp_readout:
        rule_pre_readout = F.tanh(self.rule_ctx_to_readout(torch.cat([rule_h_t, rule_ctx], dim=1)))
        word_pre_readout = F.tanh(self.word_ctx_to_readout(torch.cat([rule_h_t, rule_ctx, word_h_t, word_ctx], dim=1)))
      else:
        inp = torch.cat([rule_h_t, rule_ctx, word_h_t, word_ctx], dim=1)
        rule_pre_readout = F.tanh(self.rule_ctx_to_readout(inp))
        word_pre_readout = F.tanh(self.word_ctx_to_readout(inp))
      
      rule_pre_readout = self.dropout(rule_pre_readout)
      word_pre_readout = self.dropout(word_pre_readout)
      
      rule_pre_readouts.append(rule_pre_readout)
      word_pre_readouts.append(word_pre_readout)

      #inp = torch.cat([rule_h_t, rule_ctx, word_h_t, word_ctx], dim=1)
      #if hasattr(self.hparams, "single_readout") and self.hparams.single_readout:
      #  pre_readout = F.tanh(self.ctx_to_readout(r_inp))
      #  pre_readout = self.dropout(pre_readout)
      #  pre_readouts.append(pre_readout)
      #else:
      #  rule_pre_readout = F.tanh(self.rule_ctx_to_readout(r_inp))
      #  word_pre_readout = F.tanh(self.word_ctx_to_readout(w_inp))
      #  
      #  rule_pre_readout = self.dropout(rule_pre_readout)
      #  word_pre_readout = self.dropout(word_pre_readout)
      #  
      #  rule_pre_readouts.append(rule_pre_readout)
      #  word_pre_readouts.append(word_pre_readout)
   
      rule_input_feed = rule_ctx
      word_input_feed = word_ctx

      rule_hidden = (rule_h_t, rule_c_t)
      word_hidden = (word_h_t, word_c_t)
      if self.hparams.parent_feed:
        all_state = torch.cat([all_state, rule_h_t], dim=1)
    if hasattr(self.hparams, "single_readout") and self.hparams.single_readout:
      readouts = self.readout(torch.stack(pre_readouts))
      logits = readouts.transpose(0, 1).contiguous()
    else:
      # [len_y, batch_size, trg_vocab_size]
      rule_readouts = self.readout(torch.stack(rule_pre_readouts))[:,:,-self.hparams.target_rule_vocab_size:]
      if hasattr(self.hparams, "rule_tanh") and self.hparams.rule_tanh > 0:
        rule_readouts = self.hparams.rule_tanh * torch.tanh(rule_readouts)
      word_readouts = self.readout(torch.stack(word_pre_readouts))[:,:,:self.hparams.target_word_vocab_size]
      if self.hparams.label_smooth > 0:
        smooth = self.hparams.label_smooth
        rule_probs = (1.0 - smooth) * F.softmax(rule_readouts, dim=2) + smooth / self.hparams.target_rule_vocab_size
        rule_readouts = torch.log(rule_probs) 
      # [batch_size, len_y, trg_vocab_size]
      logits = torch.cat([word_readouts, rule_readouts], dim=2).transpose(0, 1).contiguous()
    score_mask = score_mask.unsqueeze(2).float().data
    mask_t = self.word_vocab_mask * (1-score_mask) + self.rule_vocab_mask * score_mask
    logits.data.masked_fill_(mask_t.byte(), -float("inf"))
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
    rule_h_t, rule_c_t = rule_hidden
    
    y_emb_tm1 = self.emb(y_tm1)
    cur_nonterm = open_nonterms[-1]
    parent_state = cur_nonterm.parent_state
   
    if hyp.y[-1] < self.hparams.target_word_vocab_size:
      # word
      if self.hparams.parent_feed:
        word_input = torch.cat([y_emb_tm1, parent_state, word_input_feed], dim=1)
      else:
        word_input = torch.cat([y_emb_tm1, rule_h_t, word_input_feed], dim=1)
      word_h_t, word_c_t = self.word_lstm_cell(word_input, word_hidden)
      if hasattr(self.hparams, "single_attn") and self.hparams.single_attn:
        word_ctx = self.attention(word_h_t, x_enc_k, x_enc)
      else:
        word_ctx = self.word_attention(word_h_t, x_enc_k, x_enc)
    else:
      word_ctx = hyp.word_ctx_tm1

    update_rule_rnn = True
    if hasattr(self.hparams, "no_word_to_rule") and self.hparams.no_word_to_rule and hyp.y[-1] < self.hparams.target_word_vocab_size:
      if hyp.y[-1] != self.hparams.eos_id:
        update_rule_rnn = False

    if update_rule_rnn:
      if self.hparams.rule_parent_feed:
        rule_input = torch.cat([y_emb_tm1, parent_state, rule_input_feed, word_h_t], dim=1)
      else:
        rule_input = torch.cat([y_emb_tm1, rule_input_feed, word_h_t], dim=1)
      rule_h_t, rule_c_t = self.rule_lstm_cell(rule_input, rule_hidden)
    #  if hasattr(self.hparams, "single_attn") and self.hparams.single_attn:
    #    rule_ctx = self.attention(rule_h_t, x_enc_k, x_enc)
    #  else:
    #    rule_ctx = self.rule_attention(rule_h_t, x_enc_k, x_enc)
    #else:
    #  rule_ctx = hyp.rule_ctx_tm1
    if hasattr(self.hparams, "single_attn") and self.hparams.single_attn:
      rule_ctx = self.attention(rule_h_t, x_enc_k, x_enc)
    else:
      rule_ctx = self.rule_attention(rule_h_t, x_enc_k, x_enc)

    mask = torch.ones(1, self.target_vocab_size).byte()
    if cur_nonterm.label == '*':
      word_index = torch.arange(self.hparams.target_word_vocab_size).long()
      mask.index_fill_(1, word_index, 0)
      #if hasattr(self.hparams, "single_inp_readout") and self.hparams.single_inp_readout:
      #  inp = torch.cat([word_h_t, word_ctx], dim=1)
      #else:
      #  inp = torch.cat([rule_h_t, rule_ctx, word_h_t, word_ctx], dim=1)
      inp = torch.cat([rule_h_t, rule_ctx, word_h_t, word_ctx], dim=1)
      if hasattr(self.hparams, "single_readout") and self.hparams.single_readout:
        word_pre_readout = F.tanh(self.ctx_to_readout(inp))
      else:
        word_pre_readout = F.tanh(self.word_ctx_to_readout(inp))
      #word_pre_readout = self.dropout(word_pre_readout)
      score_t = self.readout(word_pre_readout)
      num_rule_index = -1
      rule_select_index = []
    else:
      rule_with_lhs = target_rule_vocab.rule_index_with_lhs(cur_nonterm.label)
      rule_select_index = []
      for i in rule_with_lhs: rule_select_index.append(i + self.hparams.target_word_vocab_size)
      num_rule_index = len(rule_with_lhs)
      rule_index = torch.arange(self.hparams.target_rule_vocab_size).long() + self.hparams.target_word_vocab_size
      mask.index_fill_(1, rule_index, 0)
      if hasattr(self.hparams, "single_inp_readout") and self.hparams.single_inp_readout:
        inp = torch.cat([rule_h_t, rule_ctx], dim=1)
      else:
        inp = torch.cat([rule_h_t, rule_ctx, word_h_t, word_ctx], dim=1)
      if hasattr(self.hparams, "single_readout") and self.hparams.single_readout:
        rule_pre_readout = F.tanh(self.ctx_to_readout(inp))  
      else:
        rule_pre_readout = F.tanh(self.rule_ctx_to_readout(inp))  
      #rule_pre_readout = self.dropout(rule_pre_readout)
      score_t = self.readout(rule_pre_readout)
    if self.hparams.cuda:
      mask = mask.cuda()
    score_t.data.masked_fill_(mask, -float("inf"))
    rule_hidden = (rule_h_t, rule_c_t)
    word_hidden = (word_h_t, word_c_t)
    return score_t, rule_hidden, word_hidden, rule_ctx, word_ctx, num_rule_index, rule_select_index

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

  def forward(self, x_train, x_mask, x_len, y_train, y_mask, y_len, score_mask, y_label=None):
    # [batch_size, x_len, d_model * 2]
    x_enc, dec_init = self.encoder(x_train, x_len)
    x_enc_k = self.enc_to_k(x_enc)
    # [batch_size, y_len-1, trg_vocab_size]
    logits = self.decoder(x_enc, x_enc_k, dec_init, x_mask, y_train, y_mask, score_mask, y_label)
    return logits

  def translate(self, x_train, target_rule_vocab, max_len=100, beam_size=5, y_label=None, poly_norm_m=0):
    hyps = []
    scores = []
    i = 0
    self.logsoftmax = nn.LogSoftmax(dim=1)
    for x in x_train:
      x = Variable(torch.LongTensor(x), volatile=True)
      if y_label:
        y = y_label[i][1:]
      else:
        y = None
      if self.hparams.cuda:
        x = x.cuda()
      hyp, nll_score = self.translate_sent(x, target_rule_vocab, max_len=max_len, beam_size=beam_size, y_label=y, poly_norm_m=poly_norm_m)
      hyp = hyp[0]
      hyps.append(hyp.y[1:])
      scores.append(sum(nll_score))
      #print(hyp.y)
      #print("trans score:", nll_score)
      #print("trans label:", y)
      i += 1
      gc.collect()
    return hyps, scores

  def translate_sent(self, x_train, target_rule_vocab, max_len=100, beam_size=5, y_label=None, poly_norm_m=1.):
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
                  open_nonterms=[OpenNonterm(label=self.hparams.root_label, 
                    parent_state=Variable(state_zeros, requires_grad=False))],
                  score=0.)]
    nll_score = []
    if y_label is not None:
      max_len = len(y_label)
    while len(completed_hyp) < beam_size and length < max_len:
      length += 1
      new_active_hyp = []
      for i, hyp in enumerate(active_hyp):
        logits, rule_hidden, word_hidden, rule_ctx, word_ctx, num_rule_index, rule_index = self.decoder.step(x_enc, 
          x_enc_k, hyp, target_rule_vocab)
        hyp.rule_hidden = rule_hidden
        hyp.word_hidden = word_hidden
        hyp.rule_ctx_tm1 = rule_ctx 
        hyp.word_ctx_tm1 = word_ctx

        rule_index = set(rule_index)
        logits = logits.view(-1)
        p_t = F.log_softmax(logits, dim=0).data
        if poly_norm_m > 0 and length > 1:
          new_hyp_scores = (hyp.score * pow(length-1, poly_norm_m) + p_t) / pow(length, poly_norm_m)
        else:
          new_hyp_scores = hyp.score + p_t
        if y_label is not None:
          top_ids = [y_label[length-1][0]]
          nll = -(p_t[top_ids[0]])
          nll_score.append(nll)
          print(logits[top_ids[0]])
        else:
          num_select = beam_size
          if num_rule_index >= 0: num_select = min(num_select, num_rule_index)
          top_ids = (-new_hyp_scores).cpu().numpy().argsort()[:num_select]
        for word_id in top_ids:
          if y_label is None and len(rule_index) > 0 and word_id not in rule_index: continue
          open_nonterms = hyp.open_nonterms[:]
          if word_id >= self.hparams.target_word_vocab_size:
            rule = target_rule_vocab[word_id]
            cur_nonterm = open_nonterms.pop()
            parent_state = hyp.rule_hidden[0]
            for c in reversed(rule.rhs):
              open_nonterms.append(OpenNonterm(label=c, parent_state=parent_state))
          else:
            if open_nonterms[-1].label != '*':
              print(open_nonterms[-1].label, word_id, new_hyp_scores[word_id])
              print(target_rule_vocab.rule_index_with_lhs(open_nonterms[-1].label))
              print(top_ids)
            assert open_nonterms[-1].label == '*'
            if word_id == self.hparams.eos_id: 
              open_nonterms.pop()
          new_hyp = TrHyp(rule_hidden=(hyp.rule_hidden[0], hyp.rule_hidden[1]), 
                    word_hidden=(hyp.word_hidden[0], hyp.word_hidden[1]),
                    y=hyp.y+[word_id], 
                    rule_ctx_tm1=hyp.rule_ctx_tm1, 
                    word_ctx_tm1=hyp.word_ctx_tm1,
                    open_nonterms=open_nonterms,
                    score=new_hyp_scores[word_id],
                    c_p=p_t[word_id])
          new_active_hyp.append(new_hyp)
      if y_label is None:
        live_hyp_num = beam_size - len(completed_hyp)
        new_active_hyp = sorted(new_active_hyp, key=lambda x:x.score, reverse=True)[:min(beam_size, live_hyp_num)]
        active_hyp = []
        for hyp  in new_active_hyp:
          if len(hyp.open_nonterms) == 0:
            #if poly_norm_m <= 0:
            #  hyp.score = hyp.score / len(hyp.y)
            completed_hyp.append(hyp)
          else:
            active_hyp.append(hyp)
      else:
        active_hyp = new_active_hyp

    if len(completed_hyp) == 0:
      completed_hyp.append(active_hyp[0])
    return sorted(completed_hyp, key=lambda x: x.score, reverse=True), nll_score

class TrHyp(object):
  def __init__(self, rule_hidden, word_hidden, y, rule_ctx_tm1, word_ctx_tm1, score, open_nonterms, c_p=0):
    self.rule_hidden = rule_hidden
    self.word_hidden = word_hidden
    # [length_y, 2], each element (index, is_word)
    self.y = y 
    self.rule_ctx_tm1 = rule_ctx_tm1
    self.word_ctx_tm1 = word_ctx_tm1
    self.score = score
    self.open_nonterms = open_nonterms
    self.c_p = c_p

class OpenNonterm(object):
  def __init__(self, label, parent_state):
    self.label = label
    self.parent_state = parent_state
