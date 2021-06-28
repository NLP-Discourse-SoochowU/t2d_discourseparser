# coding: UTF-8

import logging
from collections import OrderedDict
from nltk import ParentedTree
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC


logger = logging.getLogger(__name__)


class SVMCommaClassifier:
    def __init__(self, connectives, candidate=",，；", seed=21):
        self.connectives = connectives
        self.candidate = candidate
        self.fet_vector = DictVectorizer()
        self.clf = LinearSVC(random_state=seed)

    def predict(self, comma_pos, parse):
        fet = self.extract_features(comma_pos, parse)
        x = self.fet_vector.transform([fet])
        return self.clf.predict(x)[0]

    def predict_many(self, x):
        fets = []
        for comma_pos, parse in x:
            fets.append(self.extract_features(comma_pos, parse))
        x = self.fet_vector.transform(fets)
        return self.clf.predict(x)

    def extract_features(self, comma_pos, parse):
        childs = list(parse.subtrees(lambda t: t.height() == 2 and t.label() != '-NONE-'))
        offset = 0
        comma = None
        comma_index = -1
        for i, child in enumerate(childs):
            if offset == comma_pos:
                comma = child
                comma_index = i
            offset += len(child[0])

        if comma is None:
            return {}

        comma_prev = []
        comma_post = []
        if comma_index > 0:
            for child in childs[comma_index-1::-1]:
                if child[0] == ',' or child[0] == '，':
                    break
                else:
                    comma_prev.append(child)
        comma_prev = comma_prev[::-1]
        for child in childs[comma_index+1:]:
            if child[0] == ',' or child[0] == '，':
                break
            else:
                comma_post.append(child)

        # extract feature
        fet = OrderedDict()
        for i, prev in enumerate(comma_prev[:3]):
            fet['F1_P_%d' % (i+1)] = prev.label()
            fet['F1_W_%d' % (i+1)] = prev[0]
        for i, prev in enumerate(comma_prev[-3:]):
            fet['F2_P_%d' % (i+1)] = prev.label()
            fet['F2_W_%d' % (i+1)] = prev[0]

        if comma_post:
            fet['F3'] = comma_post[0].label()
            fet['F4'] = comma_post[0][0]

        for node in comma_prev:
            if node[0] in self.connectives:
                fet['F5_1'] = node[0]
        for node in comma_post:
            if node[0] in self.connectives:
                fet['F5_2'] = node[0]

        lsibling = comma.left_sibling()
        rsibling = comma.right_sibling()
        while isinstance(lsibling, ParentedTree) and lsibling.label() == '-NONE-':
            lsibling = lsibling.left_sibling()
        while isinstance(rsibling, ParentedTree) and rsibling.label() == '-NONE-':
            rsibling = rsibling.right_sibling()

        if lsibling:
            fet['F6'] = lsibling.label()
        if rsibling:
            fet['F7'] = rsibling.label()
        if lsibling and rsibling:
            fet['F8'] = '%s_%s' % (fet['F6'], fet['F7'])
            fet['F9'] = '%s_%s_%s' % (fet['F6'], comma.parent().label(), fet['F7'])

        for node in comma_prev:
            if node.label().startswith('VC'):
                fet['F10_1'] = 'True'
            if node.label().startswith('VA'):
                fet['F10_2'] = 'True'
            if node.label().startswith('VE'):
                fet['F10_3'] = 'True'
            if node.label().startswith('VV'):
                fet['F10_4'] = 'True'
            if node.label().startswith('CS'):
                fet['F10_5'] = 'True'
        for node in comma_post:
            if node.label().startswith('VC'):
                fet['F11_1'] = 'True'
            if node.label().startswith('VA'):
                fet['F11_2'] = 'True'
            if node.label().startswith('VE'):
                fet['F11_3'] = 'True'
            if node.label().startswith('VV'):
                fet['F11_4'] = 'True'
            if node.label().startswith('CS'):
                fet['F11_5'] = 'True'

        pcomma = comma.parent()
        if 'F9' in fet and fet['F9'] == 'IP_IP_IP':
            fet['F12'] = 'True'
        if parse.height() - pcomma.height() == 1:
            fet['F13'] = 'True'
        if 'F12' in fet and fet['F12'] and 'F13' in fet and fet['F13']:
            fet['F14'] = 'True'

        punct = []
        for child in childs:
            if child[0] in ',.?!，。？！':
                punct.append(child[0])
        fet['F15'] = '_'.join(punct)

        pre_len = len(''.join([node[0] for node in comma_prev]))
        post_len = len(''.join(node[0] for node in comma_post))
        if pre_len < 5:
            fet['F16'] = 'True'
        if abs(pre_len - post_len) > 7:
            fet['F17'] = 'True'

        comma_dept = 0
        tmp_node = comma
        while tmp_node.parent() and tmp_node.parent() is not parse:
            comma_dept += 1
            tmp_node = tmp_node.parent()
        fet['F18'] = comma_dept
        del tmp_node

        if pcomma and pcomma.label().startswith('NP'):
            fet['F19'] = 'True'
        if isinstance(lsibling, ParentedTree) and lsibling.label().startswith('NP'):
            fet['F20'] = 'True'
        if isinstance(rsibling, ParentedTree) and rsibling.label().startswith('NP'):
            fet['F21'] = 'True'

        if len(comma_prev) >= 2:
            fet['F22'] = comma_prev[0].label() + '_' + comma_prev[-1].label()
            fet['F23'] = comma_prev[0][0] + '_' + comma_prev[-1][0]

        comma_prev_set = set([(node.label(), node[0]) for node in comma_prev if node.label() != 'PU'])
        comma_post_set = set([(node.label(), node[0]) for node in comma_post if node.label() != 'PU'])
        if comma_prev_set & comma_post_set:
            fet['F24'] = list(comma_prev_set & comma_post_set)[0][0]
        return fet
