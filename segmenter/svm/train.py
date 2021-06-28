# coding: UTF-8
import argparse
import logging
import random
import re
import pickle
from itertools import chain
import numpy as np
from nltk.tree import ParentedTree
from tqdm import tqdm

from dataset import CDTB
from segmenter.svm.model import SVMCommaClassifier
from structure.nodes import node_type_filter, EDU, Sentence
from sklearn.metrics import classification_report
from util.berkely import BerkeleyParser

parser = BerkeleyParser()
logger = logging.getLogger("train svm segmenter")


def gen_instances(dataset, parses, model):
    instances = []
    labels = []
    candidate_re = re.compile("[%s]" % model.candidate)
    for paragraph in tqdm(chain(*dataset)):
        root = paragraph.root_relation()
        if root:
            sentences = list(root.iterfind(filter=node_type_filter(Sentence)))
            # 分割点两边的偏移量
            for sentence in sentences:
                segments = set()  # 分割点两侧的偏移量
                candidates = set()  # 候选分割词的偏移量
                edus = list(sentence.iterfind(filter=node_type_filter(EDU)))
                offset = 0
                for edu in edus:
                    segments.add(offset)
                    segments.add(offset+len(edu.text)-1)
                    offset += len(edu.text)
                # convert tree in parented tree for feature extraction
                parse = ParentedTree.fromstring(parser.parse(sentence.text).pformat())
                for m in candidate_re.finditer(sentence.text):
                    candidate = m.start()
                    instances.append(model.extract_features(candidate, parse))
                    labels.append(1 if candidate in segments else 0)
    return instances, labels


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dataset
    cdtb = CDTB(args.data, "TRAIN", "VALIDATE", "TEST", ctb_dir=args.ctb_dir, preprocess=True, cache_dir=args.cache_dir)
    # load connectives
    with open(args.connectives, "r", encoding="UTF-8") as connective_fd:
        connectives = connective_fd.read().split()
    # build model
    model = SVMCommaClassifier(connectives, seed=args.seed)
    # gen trainning instances
    feats, labels = gen_instances(cdtb.train, cdtb.ctb, model)

    # train model
    vect = model.fet_vector.fit_transform(feats)
    model.clf.fit(vect, labels)
    # validate
    feats_eval, labels_eval = gen_instances(cdtb.validate, cdtb.ctb, model)
    vect_eval = model.fet_vector.transform(feats_eval)
    pred_eval = model.clf.predict(vect_eval)
    logger.info("validate score:")
    logger.info("\n" + classification_report(labels_eval, pred_eval))
    # test
    feats_test, labels_test = gen_instances(cdtb.test, cdtb.ctb, model)
    vect_test = model.fet_vector.transform(feats_test)
    pred_test = model.clf.predict(vect_test)
    logger.info("test score:")
    logger.info("\n" + classification_report(labels_test, pred_test))

    # save
    logger.info("save model to %s" % args.model_save)
    with open(args.model_save, "wb+") as model_fd:
        pickle.dump(model, model_fd)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    # dataset parameters
    arg_parser.add_argument("data")
    arg_parser.add_argument("--ctb_dir")
    arg_parser.add_argument("--cache_dir")
    arg_parser.add_argument("-connectives", required=True)
    arg_parser.add_argument("--seed", default=21, type=int)
    arg_parser.add_argument("-model_save", required=True)

    main(arg_parser.parse_args())
