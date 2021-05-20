# coding: UTF-8

from interface import ParserI
from structure import Paragraph, EDU, TEXT, Relation, rev_relationmap
from treebuilder.shiftreduce.model import SHIFT, REDUCE
from collections import deque


INF = 1e8


class ShiftReduceParser(ParserI):
    def __init__(self, model):
        self.model = model
        model.eval()

    def parse(self, para: Paragraph) -> Paragraph:
        edus = []
        for edu in para.edus():
            edu_copy = EDU([TEXT(edu.text)])
            setattr(edu_copy, "words", edu.words)
            setattr(edu_copy, "tags", edu.tags)
            edus.append(edu_copy)
        if len(edus) < 2:
            return para

        trans_probs = []
        state = self.init_state(edus)
        while not self.terminate(state):
            logits = self.model(state)
            valid = self.valid_trans(state)
            for i, (trans, _, _) in enumerate(self.model.trans_label.id2label):
                if trans not in valid:
                    logits[i] = -INF
            probs = logits.softmax(dim=0)
            trans_probs.append(probs)
            next_trans, _, _ = self.model.trans_label.id2label[probs.argmax(dim=0)]
            if next_trans == SHIFT:
                state = self.model.shift(state)
            elif next_trans == REDUCE:
                state = self.model.reduce(state)
            else:
                raise ValueError("unexpected transition occured")
        parsed = self.build_tree(edus, trans_probs)
        return parsed

    def build_tree(self, edus, trans_probs):
        buffer = deque(edus)
        stack = []
        for prob in trans_probs:
            trans, nuclear, ftype = self.model.trans_label.id2label[prob.argmax()]
            ctype = rev_relationmap[ftype] if ftype is not None else None
            if trans == SHIFT:
                stack.append(buffer.popleft())
            elif trans == REDUCE:
                right = stack.pop()
                left = stack.pop()
                comp = Relation([left, right], nuclear=nuclear, ftype=ftype, ctype=ctype)
                stack.append(comp)
        assert len(stack) == 1
        return Paragraph([stack[0]])

    def init_state(self, edus):
        word_ids = [[self.model.word_vocab[word] for word in edu.words] for edu in edus]
        pos_ids = [[self.model.pos_vocab[pos] for pos in edu.tags] for edu in edus]
        state = self.model.init_state(word_ids, pos_ids)
        return state

    def valid_trans(self, state):
        valid = []
        if len(state.buffer) > 2:
            valid.append(SHIFT)
        if len(state.stack) >= 4:
            valid.append(REDUCE)
        return valid

    def terminate(self, state):
        return not self.valid_trans(state)
