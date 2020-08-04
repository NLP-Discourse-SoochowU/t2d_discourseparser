# coding: UTF-8
from structure.nodes import Paragraph, Relation, rev_relationmap


class PartitionPtrParser:
    def __init__(self, model):
        self.model = model

    def parse(self, edus, ret_session=False):
        # TODO implement beam search
        session = self.model.init_session(edus)
        while not session.terminate():
            split_scores, nucs_score, rels_score, state = self.model(session)
            split = split_scores.argmax()
            nuclear_id = nucs_score[split].argmax()
            nuclear = self.model.nuc_label.id2label[nuclear_id]
            relation_id = rels_score[split].argmax()
            relation = self.model.rel_label.id2label[relation_id]
            session = session.forward(split_scores, state, split, nuclear, relation)
        # build tree by splits (left, split, right)
        root_relation = self.build_tree(edus, session.splits[:], session.nuclears[:], session.relations[:])
        discourse = Paragraph([root_relation])
        if ret_session:
            return discourse, session
        else:
            return discourse

    def build_tree(self, edus, splits, nuclears, relations):
        left, split, right = splits.pop(0)
        nuclear = nuclears.pop(0)
        ftype = relations.pop(0)
        ctype = rev_relationmap[ftype]
        if split - left == 1:
            left_node = edus[left]
        else:
            left_node = self.build_tree(edus, splits, nuclears, relations)

        if right - split == 1:
            right_node = edus[split]
        else:
            right_node = self.build_tree(edus, splits, nuclears, relations)

        relation = Relation([left_node, right_node], nuclear=nuclear, ftype=ftype, ctype=ctype)
        return relation
