# coding: UTF-8
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class MaskedGRU(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaskedGRU, self).__init__()
        self.rnn = nn.GRU(batch_first=True, *args, **kwargs)
        self.hidden_size = self.rnn.hidden_size

    def forward(self, padded, lengths, initial_state=None):
        # [batch*edu]
        zero_mask = lengths != 0
        lengths[lengths == 0] += 1  # in case zero length instance
        _, indices = lengths.sort(descending=True)
        _, rev_indices = indices.sort()

        # [batch*edu, max_word_seqlen, embedding]
        padded_sorted = padded[indices]
        lengths_sorted = lengths[indices]
        padded_packed = pack_padded_sequence(padded_sorted, lengths_sorted, batch_first=True)
        self.rnn.flatten_parameters()
        outputs_sorted_packed, hidden_sorted = self.rnn(padded_packed, initial_state)
        # [batch*edu, max_word_seqlen, ]
        outputs_sorted, _ = pad_packed_sequence(outputs_sorted_packed, batch_first=True)
        # [batch*edu, max_word_seqlen, output_size]
        outputs = outputs_sorted[rev_indices]
        # [batch*edu, output_size]
        hidden = hidden_sorted.transpose(1, 0).contiguous().view(outputs.size(0), -1)[rev_indices]

        outputs = outputs * zero_mask.view(-1, 1, 1).float()
        hidden = hidden * zero_mask.view(-1, 1).float()
        return outputs, hidden


class BiGRUEDUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRUEDUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = MaskedGRU(input_size, hidden_size//2, bidirectional=True)
        self.token_scorer = nn.Linear(hidden_size, 1)
        self.output_size = hidden_size

    def forward(self, inputs, masks):
        lengths = masks.sum(-1)
        outputs, hidden = self.rnn(inputs, lengths)
        token_score = self.token_scorer(outputs).squeeze(-1)
        token_score[masks == 0] = -1e8
        token_score = token_score.softmax(dim=-1) * masks.float()
        weighted_sum = (outputs * token_score.unsqueeze(-1)).sum(-2)
        return hidden + weighted_sum


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size
        self.input_dense = nn.Linear(input_size, hidden_size)
        self.edu_rnn = MaskedGRU(hidden_size, hidden_size//2, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.split_rnn = MaskedGRU(hidden_size, hidden_size//2, bidirectional=True)

    def forward(self, inputs, masks):
        inputs = self.input_dense(inputs)
        # edu rnn
        edus, _ = self.edu_rnn(inputs, masks.sum(-1))
        edus = inputs + self.dropout(edus)
        # cnn
        edus = edus.transpose(-2, -1)
        splits = self.conv(edus).transpose(-2, -1)
        masks = torch.cat([(masks.sum(-1, keepdim=True) > 0).type_as(masks), masks], dim=1)
        lengths = masks.sum(-1)
        # split rnn
        outputs, hidden = self.split_rnn(splits, lengths)
        outputs = splits + self.dropout(outputs)
        return outputs, masks, hidden


class Decoder(nn.Module):
    def __init__(self, inputs_size, hidden_size):
        super(Decoder, self).__init__()
        self.input_dense = nn.Linear(inputs_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_size = hidden_size

    def forward(self, input, state):
        return self.run_step(input, state)

    def run_batch(self, inputs, init_states, masks):
        inputs = self.input_dense(inputs) * masks.unsqueeze(-1).float()
        outputs, _ = self.rnn(inputs, init_states.unsqueeze(0))
        outputs = outputs * masks.unsqueeze(-1).float()
        return outputs

    def run_step(self, input, state):
        input = self.input_dense(input)
        self.rnn.flatten_parameters()
        output, state = self.rnn(input, state)
        return output, state


class BiaffineAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, num_labels, hidden_size):
        super(BiaffineAttention, self).__init__()
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.e_mlp = nn.Sequential(
            nn.Linear(encoder_size, hidden_size),
            nn.ReLU()
        )
        self.d_mlp = nn.Sequential(
            nn.Linear(decoder_size, hidden_size),
            nn.ReLU()
        )
        self.W_e = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
        self.W_d = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=torch.float))
        self.U = nn.Parameter(torch.empty(num_labels, hidden_size, hidden_size, dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(num_labels, 1, 1, dtype=torch.float))
        nn.init.xavier_normal_(self.W_e)
        nn.init.xavier_normal_(self.W_d)
        nn.init.xavier_normal_(self.U)

    def forward(self, e_outputs, d_outputs):
        # e_outputs [batch, length_encoder, encoder_size]
        # d_outputs [batch, length_decoder, decoder_size]

        # [batch, length_encoder, hidden_size]
        e_outputs = self.e_mlp(e_outputs)
        # [batch, length_encoder, hidden_size]
        d_outputs = self.d_mlp(d_outputs)

        # [batch, num_labels, 1, length_encoder]
        out_e = (self.W_e @ e_outputs.transpose(1, 2)).unsqueeze(2)
        # [batch, num_labels, length_decoder, 1]
        out_d = (self.W_d @ d_outputs.transpose(1, 2)).unsqueeze(3)

        # [batch, 1, length_decoder, hidden_size] @ [num_labels, hidden_size, hidden_size]
        # [batch, num_labels, length_decoder, hidden_size]
        out_u = d_outputs.unsqueeze(1) @ self.U
        # [batch, num_labels, length_decoder, hidden_size] * [batch, 1, hidden_size, length_encoder]
        # [batch, num_labels, length_decoder, length_encoder]
        out_u = out_u @ e_outputs.unsqueeze(1).transpose(2, 3)
        # [batch, length_decoder, length_encoder, num_labels]
        out = (out_e + out_d + out_u + self.b).permute(0, 2, 3, 1)
        return out


class SplitAttention(nn.Module):
    def __init__(self, encoder_size, decoder_size, hidden_size):
        super(SplitAttention, self).__init__()
        self.biaffine = BiaffineAttention(encoder_size, decoder_size, 1, hidden_size)

    def forward(self, e_outputs, d_outputs, masks):
        biaffine = self.biaffine(e_outputs, d_outputs)
        attn = biaffine.squeeze(-1)
        attn[masks == 0] = -1e8
        return attn


class PartitionPtr(nn.Module):
    def __init__(self, hidden_size, dropout, word_vocab, pos_vocab, nuc_label, rel_label,
                 pretrained=None, w2v_size=None, w2v_freeze=False, pos_size=30,
                 split_mlp_size=32, nuc_mlp_size=128, rel_mlp_size=128,
                 use_gpu=False):
        super(PartitionPtr, self).__init__()
        self.use_gpu = use_gpu
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.nuc_label = nuc_label
        self.rel_label = rel_label
        self.word_emb = word_vocab.embedding(pretrained=pretrained, dim=w2v_size, freeze=w2v_freeze, use_gpu=use_gpu)
        self.w2v_size = self.word_emb.weight.shape[-1]
        self.pos_emb = pos_vocab.embedding(dim=pos_size, use_gpu=use_gpu)
        self.pos_size = pos_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout

        # component
        self.edu_encoder = BiGRUEDUEncoder(self.w2v_size+self.pos_size, hidden_size)
        self.encoder = Encoder(self.edu_encoder.output_size, hidden_size, dropout)
        self.context_dense = nn.Linear(self.encoder.output_size, hidden_size)
        self.decoder = Decoder(self.encoder.output_size*2, hidden_size)
        self.split_attention = SplitAttention(self.encoder.output_size, self.decoder.output_size, split_mlp_size)
        self.nuc_classifier = BiaffineAttention(self.encoder.output_size, self.decoder.output_size, len(self.nuc_label),
                                                nuc_mlp_size)
        self.rel_classifier = BiaffineAttention(self.encoder.output_size, self.decoder.output_size, len(self.rel_label),
                                                rel_mlp_size)

    def forward(self, left, right, memory, state):
        return self.decode(left, right, memory, state)

    def decode(self, left, right, memory, state):
        d_input = torch.cat([memory[0, left], memory[0, right]]).view(1, 1, -1)
        d_output, state = self.decoder(d_input, state)
        masks = torch.zeros(1, 1, memory.size(1), dtype=torch.uint8)
        masks[0, 0, left+1:right] = 1
        if self.use_gpu:
            masks = masks.cuda()
        split_scores = self.split_attention(memory, d_output, masks)
        split_scores = split_scores.softmax(dim=-1)
        nucs_score = self.nuc_classifier(memory, d_output).softmax(dim=-1) * masks.unsqueeze(-1).float()
        rels_score = self.rel_classifier(memory, d_output).softmax(dim=-1) * masks.unsqueeze(-1).float()
        split_scores = split_scores[0, 0].cpu().detach().numpy()
        nucs_score = nucs_score[0, 0].cpu().detach().numpy()
        rels_score = rels_score[0, 0].cpu().detach().numpy()
        return split_scores, nucs_score, rels_score, state

    def encode_edus(self, e_inputs):
        e_input_words, e_input_poses, e_masks = e_inputs
        batch_size, max_edu_seqlen, max_word_seqlen = e_input_words.size()
        # [batch_size, max_edu_seqlen, max_word_seqlen, embedding]
        word_embedd = self.word_emb(e_input_words)
        pos_embedd = self.pos_emb(e_input_poses)
        concat_embedd = torch.cat([word_embedd, pos_embedd], dim=-1) * e_masks.unsqueeze(-1).float()
        # encode edu
        # [batch_size*max_edu_seqlen, max_word_seqlen, embedding]
        inputs = concat_embedd.view(batch_size*max_edu_seqlen, max_word_seqlen, -1)
        # [batch_size*max_edu_seqlen, max_word_seqlen]
        masks = e_masks.view(batch_size*max_edu_seqlen, max_word_seqlen)
        edu_encoded = self.edu_encoder(inputs, masks)
        # [batch_size, max_edu_seqlen, edu_encoder_output_size]
        edu_encoded = edu_encoded.view(batch_size, max_edu_seqlen, self.edu_encoder.output_size)
        e_masks = (e_masks.sum(-1) > 0).int()
        return edu_encoded, e_masks

    def _decode_batch(self, e_outputs, e_contexts, d_inputs):
        d_inputs_indices, d_masks = d_inputs
        d_outputs_masks = (d_masks.sum(-1) > 0).type_as(d_masks)

        d_init_states = self.context_dense(e_contexts)

        d_inputs = e_outputs[torch.arange(e_outputs.size(0)), d_inputs_indices.permute(2, 1, 0)].permute(2, 1, 0, 3)
        d_inputs = d_inputs.contiguous().view(d_inputs.size(0), d_inputs.size(1), -1)
        d_inputs = d_inputs * d_outputs_masks.unsqueeze(-1).float()

        d_outputs = self.decoder.run_batch(d_inputs, d_init_states, d_outputs_masks)
        return d_outputs, d_outputs_masks, d_masks

    def loss(self, e_inputs, d_inputs, grounds):
        e_inputs, e_masks = self.encode_edus(e_inputs)
        e_outputs, e_outputs_masks, e_contexts = self.encoder(e_inputs, e_masks)
        d_outputs, d_outputs_masks, d_masks = self._decode_batch(e_outputs, e_contexts, d_inputs)

        splits_ground, nucs_ground, rels_ground = grounds
        # split loss
        splits_attn = self.split_attention(e_outputs, d_outputs, d_masks)
        splits_predict = splits_attn.log_softmax(dim=2)
        splits_ground = splits_ground.view(-1)
        splits_predict = splits_predict.view(splits_ground.size(0), -1)
        splits_masks = d_outputs_masks.view(-1).float()
        splits_loss = F.nll_loss(splits_predict, splits_ground, reduction="none")
        splits_loss = (splits_loss * splits_masks).sum() / splits_masks.sum()
        # nuclear loss
        nucs_score = self.nuc_classifier(e_outputs, d_outputs)
        nucs_score = nucs_score.log_softmax(dim=-1) * d_masks.unsqueeze(-1).float()
        nucs_score = nucs_score.view(nucs_score.size(0)*nucs_score.size(1), nucs_score.size(2), nucs_score.size(3))
        target_nucs_score = nucs_score[torch.arange(nucs_score.size(0)), splits_ground]
        target_nucs_ground = nucs_ground.view(-1)
        nucs_loss = F.nll_loss(target_nucs_score, target_nucs_ground)

        # relation loss
        rels_score = self.rel_classifier(e_outputs, d_outputs)
        rels_score = rels_score.log_softmax(dim=-1) * d_masks.unsqueeze(-1).float()
        rels_score = rels_score.view(rels_score.size(0)*rels_score.size(1), rels_score.size(2), rels_score.size(3))
        target_rels_score = rels_score[torch.arange(rels_score.size(0)), splits_ground]
        target_rels_ground = rels_ground.view(-1)
        rels_loss = F.nll_loss(target_rels_score, target_rels_ground)

        return splits_loss, nucs_loss, rels_loss
