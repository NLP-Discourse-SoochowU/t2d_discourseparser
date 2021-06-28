# coding: UTF-8
from typing import List
from interface import SegmenterI
from nltk import ParentedTree
from structure.nodes import EDU, TEXT, Sentence, Paragraph
from util.berkely import BerkeleyParser


class SVMSegmenter(SegmenterI):
    def __init__(self, model):
        self._eos = ['！', '。', '？']
        self._pairs = {'“': "”", "「": "」"}
        self.model = model
        self.candidate = model.candidate
        self.parser = BerkeleyParser()

    def cut(self, text):
        sentences = self.cut_sent(text)
        for i, sent in enumerate(sentences):
            sentences[i] = Sentence(self.cut_edu(sent))
        return Paragraph(sentences)

    def cut_sent(self, text: str, sid=None) -> List[Sentence]:
        last_cut = 0
        sentences = []
        for i in range(0, len(text)-1):
            if text[i] in self._eos:
                sentences.append(Sentence([TEXT(text[last_cut: i+1])]))
                last_cut = i + 1
        if last_cut < len(text)-1:
            sentences.append(Sentence([TEXT(text[last_cut:])]))
        return sentences

    def cut_edu(self, sent: Sentence) -> List[EDU]:
        if not hasattr(sent, "parse"):
            print(sent.text)
            parse = self.parser.parse(sent.text)
        else:
            parse = getattr(sent, "parse")
        parse = ParentedTree.fromstring(parse.pformat())
        children = list(parse.subtrees(lambda t: t.height() == 2 and t.label() != '-NONE-'))
        edus = []
        last_edu_words = []
        last_edu_tags = []
        offset = 0
        for child in children:
            if child[0] == '-LRB-':
                child[0] = '('
            if child[0] == '-RRB-':
                child[0] = ')'
            last_edu_words.append(child[0])
            last_edu_tags.append(child.label())
            if child[0] in self._eos or (child[0] in self.candidate and self.model.predict(offset, parse)):
                text = "".join(last_edu_words)
                edu = EDU([TEXT(text)])
                setattr(edu, "words", last_edu_words)
                setattr(edu, "tags", last_edu_tags)
                edus.append(edu)
                last_edu_words = []
                last_edu_tags = []
            offset += len(child[0])
        if last_edu_words:
            text = "".join(last_edu_words)
            edu = EDU([TEXT(text)])
            setattr(edu, "words", last_edu_words)
            setattr(edu, "tags", last_edu_tags)
            edus.append(edu)
        return edus
