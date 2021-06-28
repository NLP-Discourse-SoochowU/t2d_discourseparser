# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNSegmenterModel(nn.Module):
    def __init__(self, hidden_size, dropout, rnn_layers, word_vocab, pos_vocab, tag_label,
                 pos_size=30, pretrained=None, w2v_size=None, w2v_freeze=False,
                 use_gpu=False):
        super(RNNSegmenterModel, self).__init__()
        self.use_gpu = use_gpu
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.tag_label = tag_label
        self.word_emb = word_vocab.embedding(pretrained=pretrained, dim=w2v_size, freeze=w2v_freeze, use_gpu=use_gpu)
        self.w2v_size = self.word_emb.weight.shape[-1]
        self.pos_emb = pos_vocab.embedding(dim=pos_size, use_gpu=use_gpu)
        self.pos_size = pos_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.rnn = nn.LSTM(self.w2v_size+self.pos_size, self.hidden_size // 2,
                           num_layers=rnn_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.tagger = nn.Linear(hidden_size, len(tag_label))

    def forward(self, word_ids, pos_ids, masks=None):
        self.rnn.flatten_parameters()
        word_emb = self.word_emb(word_ids)
        pos_emb = self.pos_emb(pos_ids)
        rnn_inputs = torch.cat([word_emb, pos_emb], dim=-1)
        if masks is not None:
            lengths = masks.sum(-1)
            rnn_inputs = rnn_inputs * masks.unsqueeze(-1).float()
            rnn_inputs_packed = pack_padded_sequence(rnn_inputs, lengths, batch_first=True)
            rnn_outputs_packed, _ = self.rnn(rnn_inputs_packed)
            rnn_outputs, _ = pad_packed_sequence(rnn_outputs_packed, batch_first=True)
        else:
            rnn_outputs, _ = self.rnn(rnn_inputs)
        tag_score = self.tagger(rnn_outputs)
        return tag_score

    def loss(self, inputs, target):
        word_ids, pos_ids, masks = inputs
        batch_size, max_seqlen = word_ids.size()
        pred = F.log_softmax(self(word_ids, pos_ids, masks), dim=-1)
        pred = pred.view(batch_size*max_seqlen, -1)
        target = target.view(-1)
        masks = masks.view(-1)
        losses = F.nll_loss(pred, target, reduction='none')
        loss = (losses * masks.float()).sum() / masks.sum().float()
        return loss
