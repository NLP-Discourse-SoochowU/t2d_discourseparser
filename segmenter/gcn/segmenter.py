# coding: UTF-8
import torch
from interface import SegmenterI
from structure import Sentence, EDU, Paragraph, TEXT
from util.berkely import BerkeleyParser
from util.ltp import LTPParser


class GCNSegmenter(SegmenterI):
    def __init__(self, model):
        self._eos = ['！', '。', '？']
        self._pairs = {'“': "”", "「": "」"}
        self.model = model
        self.model.eval()
        self.parser = BerkeleyParser()
        self.dep_parser = LTPParser()

    def cut(self, text):
        sentences = self.cut_sent(text)
        for i, sent in enumerate(sentences):
            sentences[i] = Sentence(self.cut_edu(sent))
        return Paragraph(sentences)

    def cut_sent(self, text, sid=None):
        last_cut = 0
        sentences = []
        for i in range(0, len(text) - 1):
            if text[i] in self._eos:
                sentences.append(Sentence([TEXT(text[last_cut: i + 1])]))
                last_cut = i + 1
        if last_cut < len(text) - 1:
            sentences.append(Sentence([TEXT(text[last_cut:])]))
        return sentences

    def cut_edu(self, sent):
        if (not hasattr(sent, "words")) or (not hasattr(sent, "tags")):
            if hasattr(sent, "parse"):
                parse = getattr(sent, "parse")
            else:
                parse = self.parser.parse(sent.text)
            children = list(parse.subtrees(lambda t: t.height() == 2 and t.label() != '-NONE-'))
            setattr(sent, "words", [child[0] for child in children])
            setattr(sent, "tags", [child.label() for child in children])

        if not hasattr(sent, "dependency"):
            dep = self.dep_parser.parse(sent.words)
            setattr(sent, "dependency", dep)

        word_ids = [self.model.word_vocab[word] for word in sent.words]
        pos_ids = [self.model.pos_vocab[pos] for pos in sent.tags]
        word_ids = torch.tensor([word_ids]).long()
        pos_ids = torch.tensor([pos_ids]).long()
        graph = torch.zeros((1, word_ids.size(1), word_ids.size(1), len(self.model.gcn_vocab)), dtype=torch.uint8)
        for i, token in enumerate(sent.dependency):
            graph[0, i, i, self.model.gcn_vocab['self']] = 1
            graph[0, i, token.head-1, self.model.gcn_vocab['head']] = 1
            graph[0, token.head-1, i, self.model.gcn_vocab['dep']] = 1
        if self.model.use_gpu:
            word_ids = word_ids.cuda()
            pos_ids = pos_ids.cuda()
            graph = graph.cuda()
        pred = self.model(word_ids, pos_ids, graph).squeeze(0)
        labels = [self.model.tag_label.id2label[t] for t in pred.argmax(-1)]

        edus = []
        last_edu_words = []
        last_edu_tags = []
        for word, pos, label in zip(sent.words, sent.tags, labels):
            last_edu_words.append(word)
            last_edu_tags.append(pos)
            if label == "B":
                text = "".join(last_edu_words)
                edu = EDU([TEXT(text)])
                setattr(edu, "words", last_edu_words)
                setattr(edu, "tags", last_edu_tags)
                edus.append(edu)
                last_edu_words = []
                last_edu_tags = []
        if last_edu_words:
            text = "".join(last_edu_words)
            edu = EDU([TEXT(text)])
            setattr(edu, "words", last_edu_words)
            setattr(edu, "tags", last_edu_tags)
            edus.append(edu)
        return edus
