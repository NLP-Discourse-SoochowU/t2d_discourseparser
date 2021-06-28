# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SyntacticGCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, bias=True):
        super(SyntacticGCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.W = nn.Parameter(torch.empty(num_labels, input_size, hidden_size, dtype=torch.float))
        nn.init.xavier_normal_(self.W)
        if bias:
            self.bias = True
            self.b = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
            nn.init.xavier_normal_(self.b)

    def forward(self, graph, nodes):
        # graph (b, n, n, l)
        # nodes (b, n, input_size)
        b, n, _ = nodes.size()
        l, input_size, hidden_size = self.num_labels, self.input_size, self.hidden_size
        # graph (b, n*l, n)
        g = graph.transpose(2, 3).float().contiguous().view(b, n*l, n)
        # x: (b, n, l*input_size)
        x = g.bmm(nodes).view(b, n, l*input_size)
        # h: (b, n, hidden_size)
        h = x.matmul(self.W.view(l*input_size, hidden_size))
        if self.bias:
            bias = (graph.float().view(b*n*n, l) @ self.b).view(b, n, n, hidden_size)
            bias = bias.sum(2)
            h = h + bias
        norm = graph.view(b, n, n*l).sum(-1).float().unsqueeze(-1) + 1e-10
        # h: (b, n, hidden_size)
        hidden = F.relu(h / norm)
        return hidden


class GCNSegmenterModel(nn.Module):
    def __init__(self, hidden_size, dropout, rnn_layers, gcn_layers, word_vocab, pos_vocab, gcn_vocab, tag_label,
                 pos_size=30, pretrained=None, w2v_size=None, w2v_freeze=False,
                 use_gpu=False):
        super(GCNSegmenterModel, self).__init__()
        self.use_gpu = use_gpu
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.gcn_vocab = gcn_vocab
        self.tag_label = tag_label
        self.word_emb = word_vocab.embedding(pretrained=pretrained, dim=w2v_size, freeze=w2v_freeze, use_gpu=use_gpu)
        self.w2v_size = self.word_emb.weight.shape[-1]
        self.pos_emb = pos_vocab.embedding(dim=pos_size, use_gpu=use_gpu)
        self.pos_size = pos_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.rnn_layers = rnn_layers
        self.rnn = nn.LSTM(self.w2v_size+self.pos_size, self.hidden_size // 2,
                           num_layers=rnn_layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.gcn_layers = gcn_layers
        self.gcns = nn.ModuleList([SyntacticGCN(hidden_size, hidden_size, len(gcn_vocab)) for _ in range(gcn_layers)])
        self.tagger = nn.Linear(hidden_size, len(tag_label))

    def forward(self, word_ids, pos_ids, graph, masks=None):
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

        for gcn in self.gcns:
            gcn_outputs = gcn(graph, rnn_outputs)
        tag_score = self.tagger(gcn_outputs)
        return tag_score

    def loss(self, inputs, target):
        word_ids, pos_ids, graph, masks = inputs
        batch_size, max_seqlen = word_ids.size()
        pred = F.log_softmax(self(word_ids, pos_ids, graph, masks), dim=-1)
        pred = pred.view(batch_size*max_seqlen, -1)
        target = target.view(-1)
        masks = masks.view(-1)
        losses = F.nll_loss(pred, target, reduction='none')
        loss = (losses * masks.float()).sum() / masks.sum().float()
        return loss
