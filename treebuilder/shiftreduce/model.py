# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


SHIFT = "SHIFT"
REDUCE = "REDUCE"


class ShiftReduceState:
    def __init__(self, stack, buffer, tracking):
        self.stack = stack
        self.buffer = buffer
        self.tracking = tracking

    def __copy__(self):
        stack = [(hs.clone(), cs.clone()) for hs, cs in self.stack]
        buffer = deque([(hb.clone(), cb.clone()) for hb, cb in self.buffer])
        h, c = self.tracking
        tracking = h.clone(), c.clone()
        return ShiftReduceState(stack, buffer, tracking)


class Reducer(nn.Module):
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.comp = nn.Linear(self.hidden_size * 3, self.hidden_size * 5)

    def forward(self, state):
        (h1, c1), (h2, c2) = state.stack[-1], state.stack[-2]
        tracking_h = state.tracking[0].view(-1)
        a, i, f1, f2, o = self.comp(torch.cat([h1, h2, tracking_h])).chunk(5)
        c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
        h = o.sigmoid() * c.tanh()
        return h, c


class MLP(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_p, num_classes):
        nn.Module.__init__(self)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers-1)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers - 1)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=dropout_p) for _ in range(num_layers - 1)])
        self.logits = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden):
        for linear, dropout, activation in zip(self.linears, self.dropouts, self.activations):
            hidden = linear(hidden)
            hidden = activation(hidden)
            hidden = dropout(hidden)
        return self.logits(hidden)


class ShiftReduceModel(nn.Module):
    def __init__(self, hidden_size, dropout, cnn_filters, word_vocab, pos_vocab, trans_label,
                 pretrained=None, w2v_size=None, w2v_freeze=False, pos_size=30, mlp_layers=1,
                 use_gpu=False):
        super(ShiftReduceModel, self).__init__()
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.trans_label = trans_label
        self.word_emb = word_vocab.embedding(pretrained=pretrained, dim=w2v_size, freeze=w2v_freeze, use_gpu=use_gpu)
        self.w2v_size = self.word_emb.weight.shape[-1]
        self.pos_emb = pos_vocab.embedding(dim=pos_size, use_gpu=use_gpu)
        self.pos_size = pos_size

        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.use_gpu = use_gpu

        # components
        cnn_input_width = self.w2v_size + self.pos_size
        unigram_filter_num, bigram_filter_num, trigram_filter_num = cnn_filters
        self.edu_unigram_cnn = nn.Conv2d(1, unigram_filter_num, (1, cnn_input_width), padding=(0, 0))
        self.edu_bigram_cnn = nn.Conv2d(1, bigram_filter_num, (2, cnn_input_width), padding=(1, 0))
        self.edu_trigram_cnn = nn.Conv2d(1, trigram_filter_num, (3, cnn_input_width), padding=(2, 0))

        self.edu_proj = nn.Linear(self.w2v_size * 2 + self.pos_size + sum(cnn_filters), self.hidden_size * 2)
        self.tracker = nn.LSTMCell(hidden_size * 3, hidden_size)
        self.reducer = Reducer(hidden_size)
        self.scorer = MLP(hidden_size, mlp_layers, dropout, len(trans_label))

    def forward(self, state):
        return self.scorer(state.tracking[0].view(-1))

    def shift(self, state):
        assert len(state.buffer) > 2
        b1 = state.buffer.popleft()
        state.stack.append(b1)
        return self.update_tracking(state)

    def reduce(self, state):
        assert len(state.stack) >= 4
        reduced = self.reducer(state)
        state.stack.pop()
        state.stack.pop()
        state.stack.append(reduced)
        return self.update_tracking(state)

    def init_state(self, edu_words, edu_poses):
        edu_encoded = self.encode_edus(edu_words, edu_poses)
        placeholder = torch.zeros(self.hidden_size)
        if self.use_gpu:
            placeholder = placeholder.cuda()
        stack = [(placeholder.clone(), placeholder.clone()),
                 (placeholder.clone(), placeholder.clone())]
        buffer = deque(edu_encoded +
                       [(placeholder.clone(), placeholder.clone()), (placeholder.clone(), placeholder.clone())])
        tracking = placeholder.clone().view(1, -1), placeholder.clone().view(1, -1)
        state = ShiftReduceState(stack, buffer, tracking)
        return self.update_tracking(state)

    def update_tracking(self, state):
        (s1, _), (s2, _) = state.stack[-1], state.stack[-2]
        b1, _ = state.buffer[0]
        cell_input = torch.cat([s1, s2, b1], dim=0).view(1, -1)
        new_tracking = self.tracker(cell_input, state.tracking)
        state.tracking = new_tracking
        return state

    def loss(self, edu_words, edu_poses, transes):
        state = self.init_state(edu_words, edu_poses)
        pred_trans_logits = []
        for trans_id in transes:
            trans, _,  _ = self.trans_label.id2label[trans_id]
            logits = self(state)
            pred_trans_logits.append(logits)
            if trans == SHIFT:
                state = self.shift(state)
            elif trans == REDUCE:
                state = self.reduce(state)
            else:
                raise ValueError("Unkown transition")

        pred = torch.stack(pred_trans_logits, dim=0)
        gold = torch.tensor(transes).long()
        if self.use_gpu:
            gold = gold.cuda()
        loss = F.cross_entropy(pred, gold)
        return loss

    def encode_edus(self, edu_words, edu_poses):
        encoded = []
        for words, poses in zip(edu_words, edu_poses):
            word_ids = torch.tensor(words or [0]).long()
            pos_ids = torch.tensor(poses or [0]).long()
            if self.use_gpu:
                word_ids = word_ids.cuda()
                pos_ids = pos_ids.cuda()
            word_embs = self.word_emb(word_ids)
            pos_embs = self.pos_emb(pos_ids)
            # basic
            w1, w_1 = word_embs[0], word_embs[-1]
            p1 = pos_embs[0]
            # cnn
            cnn_input = torch.cat([word_embs, pos_embs], dim=1)
            cnn_input = cnn_input.view(1, 1, cnn_input.size(0), cnn_input.size(1))
            unigram_output = F.relu(self.edu_unigram_cnn(cnn_input)).squeeze(-1)
            unigram_feats = F.max_pool1d(unigram_output, kernel_size=unigram_output.size(2)).view(-1)
            bigram_output = F.relu(self.edu_bigram_cnn(cnn_input)).squeeze(-1)
            bigram_feats = F.max_pool1d(bigram_output, kernel_size=bigram_output.size(2)).view(-1)
            trigram_output = F.relu(self.edu_trigram_cnn(cnn_input)).squeeze(-1)
            trigram_feats = F.max_pool1d(trigram_output, kernel_size=trigram_output.size(2)).view(-1)
            cnn_feats = torch.cat([unigram_feats, bigram_feats, trigram_feats], dim=0)
            # proj
            h, c = self.edu_proj(torch.cat([w1, w_1, p1, cnn_feats], dim=0)).chunk(2)
            encoded.append((h, c))
        return encoded
