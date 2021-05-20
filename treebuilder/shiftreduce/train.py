# coding: UTF-8
import logging
import argparse
import random
import copy
import torch
import torch.optim as optim
import numpy as np
from itertools import chain
from dataset import CDTB
from collections import Counter
from structure import EDU, Sentence, Relation, node_type_filter, TEXT, Paragraph
from structure.vocab import Label, Vocab
from treebuilder.shiftreduce.model import ShiftReduceModel, SHIFT, REDUCE
from treebuilder.shiftreduce.parser import ShiftReduceParser
from util.eval import parse_eval, gen_parse_report


def oracle(tree):
    if tree.root_relation() is None:
        raise ValueError("Can not conduct transitions from forest")

    def _oracle(root):
        trans, children = [], []
        if isinstance(root, EDU):
            trans.append((SHIFT, None, None))
            children.append(root)
        elif isinstance(root, Sentence):
            for node in root:
                _trans, _children = _oracle(node)
                trans.extend(_trans)
                children.extend(_children)
        elif isinstance(root, Relation):
            rel_children = []
            for node in root:
                _trans, _children = _oracle(node)
                trans.extend(_trans)
                rel_children.extend(_children)
            while len(rel_children) > 1:
                rel_children.pop()
                trans.append((REDUCE, root.nuclear, root.ftype))
            children.append(root)
        else:
            raise ValueError("unhandle node type %s" % repr(type(root)))
        return trans, children

    transitions, _ = _oracle(tree.root_relation())
    return transitions


def gen_instances(trees):
    instances = []
    for tree in trees:
        root = tree.root_relation()
        if root is not None:
            words = []
            poses = []
            for edu in root.iterfind(node_type_filter(EDU)):
                words.append(edu.words)
                poses.append(edu.tags)
            trans = oracle(tree)
            instances.append((words, poses, trans))
    return instances


def build_vocab(instances):
    words_counter = Counter()
    poses_counter = Counter()
    trans_counter = Counter()
    for words, poses, trans in instances:
        words_counter.update(chain(*words))
        poses_counter.update(chain(*poses))
        trans_counter.update(trans)
    word_vocab = Vocab("word", words_counter)
    pos_vocab = Vocab("part of speech", poses_counter)
    trans_label = Label("transition", trans_counter)
    return word_vocab, pos_vocab, trans_label


def numericalize(instances, word_vocab, pos_vocab, trans_label):
    ids = []
    for edu_words, edu_poses, transes in instances:
        word_ids = [[word_vocab[word] for word in edu] for edu in edu_words]
        pos_ids = [[pos_vocab[pos] for pos in edu] for edu in edu_poses]
        trans_ids = [trans_label[trans] for trans in transes]
        ids.append((word_ids, pos_ids, trans_ids))
    return ids


def gen_batch(dataset, batch_size):
    offset = 0
    while offset < len(dataset):
        _offset = offset + batch_size if offset + batch_size < len(dataset) else len(dataset)
        yield dataset[offset: _offset]
        offset = _offset


def parse_and_eval(dataset, model):
    parser = ShiftReduceParser(model)
    golds = list(filter(lambda d: d.root_relation(), chain(*dataset)))
    num_instances = len(golds)
    strips = []
    for paragraph in golds:
        edus = []
        for edu in paragraph.edus():
            edu_copy = EDU([TEXT(edu.text)])
            setattr(edu_copy, "words", edu.words)
            setattr(edu_copy, "tags", edu.tags)
            edus.append(edu_copy)
        strips.append(Paragraph(edus))

    parses = []
    for strip in strips:
        parses.append(parser.parse(strip))
    return num_instances, parse_eval(parses, golds)


def model_score(scores):
    eval_score = sum(score[2] for score in scores)
    return eval_score


def main(args):
    # set seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # load dataset
    cdtb = CDTB(args.data, "TRAIN", "VALIDATE", "TEST", ctb_dir=args.ctb_dir, preprocess=True, cache_dir=args.cache_dir)

    trainset = gen_instances(chain(*cdtb.train))
    logging.info("generate %d instances from trainset" % len(trainset))
    word_vocab, pos_vocab, trans_label = build_vocab(trainset)
    trainset = numericalize(trainset, word_vocab, pos_vocab, trans_label)

    model = ShiftReduceModel(hidden_size=args.hidden_size, dropout=args.dropout, cnn_filters=args.cnn_filters,
                             word_vocab=word_vocab, pos_vocab=pos_vocab, trans_label=trans_label,
                             pretrained=args.pretrained, w2v_size=args.w2v_size, w2v_freeze=args.w2v_freeze,
                             pos_size=args.pos_size, mlp_layers=args.mlp_layers,
                             use_gpu=args.use_gpu)
    if args.use_gpu:
        model.cuda()
    logging.info("model:\n" + str(model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    dataset = np.array(trainset)
    niter = 0
    best_model = None
    best_model_score = 0.
    for nepoch in range(1, args.epoch + 1):
        np.random.shuffle(dataset)
        batch_iter = gen_batch(dataset, args.batch_size)
        for nbatch, batch in enumerate(batch_iter):
            niter += 1
            model.train()
            optimizer.zero_grad()
            loss = 0.
            for word_ids, pos_ids, trans_ids in batch:
                batch_loss = model.loss(word_ids, pos_ids, trans_ids)
                loss += batch_loss
            loss = loss / len(batch)
            loss.backward()
            optimizer.step()
            if niter % args.log_every == 0:
                logging.info("[iter %-6d]epoch: %-3d, batch %-5d, train loss %.5f" %
                             (niter, nepoch, nbatch, loss.item()))

            if niter % args.validate_every == 0:
                model.eval()
                num_instances, validate_scores = parse_and_eval(cdtb.validate, model)
                logging.info("validation on %d instances" % num_instances)
                logging.info(gen_parse_report(*validate_scores))
                new_model_score = model_score(validate_scores)
                if new_model_score > best_model_score:
                    # test on testset with new best model
                    best_model_score = new_model_score
                    best_model = copy.deepcopy(model)
                    logging.info("test on new best model")
                    num_instances, test_scores = parse_and_eval(cdtb.test, best_model)
                    logging.info("test on %d instances" % num_instances)
                    logging.info(gen_parse_report(*test_scores))
    if best_model:
        # evaluation and save best model
        logging.info("final test result")
        num_instances, test_scores = parse_and_eval(cdtb.test, best_model)
        logging.info("test on %d instances" % num_instances)
        logging.info(gen_parse_report(*test_scores))
        logging.info("save best model to %s" % args.model_save)
        with open(args.model_save, "wb+") as model_fd:
            torch.save(best_model, model_fd)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()

    # dataset parameters
    arg_parser.add_argument("data")
    arg_parser.add_argument("--ctb_dir")
    arg_parser.add_argument("--cache_dir")

    # model parameters
    arg_parser.add_argument("-hidden_size", default=256, type=int)
    arg_parser.add_argument("-dropout", default=0.33, type=float)
    w2v_group = arg_parser.add_mutually_exclusive_group(required=True)
    w2v_group.add_argument("-pretrained")
    w2v_group.add_argument("-w2v_size", type=int)
    arg_parser.add_argument("-pos_size", default=30, type=int)
    arg_parser.add_argument("--w2v_freeze", dest="w2v_freeze", action="store_true")
    arg_parser.add_argument("-cnn_filters", nargs=3, default=[60, 30, 10], type=int)
    arg_parser.add_argument("-mlp_layers", default=2, type=int)
    arg_parser.set_defaults(w2v_freeze=False)

    # train parameters
    arg_parser.add_argument("--seed", default=21, type=int)
    arg_parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    arg_parser.set_defaults(use_gpu=False)
    arg_parser.add_argument("--epoch", default=20, type=int)
    arg_parser.add_argument("--batch_size", default=32)
    arg_parser.add_argument("-lr", default=0.001, type=float)
    arg_parser.add_argument("-l2", default=0.00001, type=float)
    arg_parser.add_argument("-log_every", default=3, type=int)
    arg_parser.add_argument("-validate_every", default=10, type=int)
    arg_parser.add_argument("-model_save", default="data/models/treebuilder.shiftreduce.model")
    main(arg_parser.parse_args())
