# coding: UTF-8
import argparse
import logging
import random
import torch
import copy
import numpy as np
from dataset import CDTB
from collections import Counter
from itertools import chain
from structure.vocab import Vocab, Label
from structure.nodes import node_type_filter, EDU, Relation, Sentence, TEXT
from treebuilder.partptr.model import PartitionPtr
from treebuilder.partptr.parser import PartitionPtrParser
import torch.optim as optim
from util.eval import evaluation_trees
from tensorboardX import SummaryWriter


def build_vocab(dataset):
    word_freq = Counter()
    pos_freq = Counter()
    nuc_freq = Counter()
    rel_freq = Counter()
    for paragraph in chain(*dataset):
        for node in paragraph.iterfind(filter=node_type_filter([EDU, Relation])):
            if isinstance(node, EDU):
                word_freq.update(node.words)
                pos_freq.update(node.tags)
            elif isinstance(node, Relation):
                nuc_freq[node.nuclear] += 1
                rel_freq[node.ftype] += 1

    word_vocab = Vocab("word", word_freq)
    pos_vocab = Vocab("part of speech", pos_freq)
    nuc_label = Label("nuclear", nuc_freq)
    rel_label = Label("relation", rel_freq)
    return word_vocab, pos_vocab, nuc_label, rel_label


def gen_decoder_data(root, edu2ids):
    # splits    s0  s1  s2  s3  s4  s5  s6
    # edus    s/  e0  e1  e2  e3  e4  e5  /s
    splits = []  # [(0, 3, 6, NS), (0, 2, 3, SN), ...]
    child_edus = []  # [edus]

    if isinstance(root, EDU):
        child_edus.append(root)
    elif isinstance(root, Sentence):
        for child in root:
            _child_edus, _splits = gen_decoder_data(child, edu2ids)
            child_edus.extend(_child_edus)
            splits.extend(_splits)
    elif isinstance(root, Relation):
        children = [gen_decoder_data(child, edu2ids) for child in root]
        if len(children) < 2:
            raise ValueError("relation node should at least 2 children")

        while children:
            left_child_edus, left_child_splits = children.pop(0)
            if children:
                last_child_edus, _ = children[-1]
                start = edu2ids[left_child_edus[0]]
                split = edu2ids[left_child_edus[-1]] + 1
                end = edu2ids[last_child_edus[-1]] + 1
                nuc = root.nuclear
                rel = root.ftype
                splits.append((start, split, end, nuc, rel))
            child_edus.extend(left_child_edus)
            splits.extend(left_child_splits)
    return child_edus, splits


def numericalize(dataset, word_vocab, pos_vocab, nuc_label, rel_label):
    instances = []
    for paragraph in filter(lambda d: d.root_relation(), chain(*dataset)):
        encoder_inputs = []
        decoder_inputs = []
        pred_splits = []
        pred_nucs = []
        pred_rels = []
        edus = list(paragraph.edus())
        for edu in edus:
            edu_word_ids = [word_vocab[word] for word in edu.words]
            edu_pos_ids = [pos_vocab[pos] for pos in edu.tags]
            encoder_inputs.append((edu_word_ids, edu_pos_ids))
        edu2ids = {edu: i for i, edu in enumerate(edus)}
        _, splits = gen_decoder_data(paragraph.root_relation(), edu2ids)
        for start, split, end, nuc, rel in splits:
            decoder_inputs.append((start, end))
            pred_splits.append(split)
            pred_nucs.append(nuc_label[nuc])
            pred_rels.append(rel_label[rel])
        instances.append((encoder_inputs, decoder_inputs, pred_splits, pred_nucs, pred_rels))
    return instances


def gen_batch_iter(instances, batch_size, use_gpu=False):
    random_instances = np.random.permutation(instances)
    num_instances = len(instances)
    offset = 0
    while offset < num_instances:
        batch = random_instances[offset: min(num_instances, offset+batch_size)]

        # find out max seqlen of edus and words of edus
        num_batch = batch.shape[0]
        max_edu_seqlen = 0
        max_word_seqlen = 0
        for encoder_inputs, decoder_inputs, pred_splits, pred_nucs, pred_rels in batch:
            max_edu_seqlen = max_edu_seqlen if max_edu_seqlen >= len(encoder_inputs) else len(encoder_inputs)
            for edu_word_ids, edu_pos_ids in encoder_inputs:
                max_word_seqlen = max_word_seqlen if max_word_seqlen >= len(edu_word_ids) else len(edu_word_ids)

        # batch to numpy
        e_input_words = np.zeros([num_batch, max_edu_seqlen, max_word_seqlen], dtype=np.long)
        e_input_poses = np.zeros([num_batch, max_edu_seqlen, max_word_seqlen], dtype=np.long)
        e_masks = np.zeros([num_batch, max_edu_seqlen, max_word_seqlen], dtype=np.uint8)

        d_inputs = np.zeros([num_batch, max_edu_seqlen-1, 2], dtype=np.long)
        d_outputs = np.zeros([num_batch, max_edu_seqlen-1], dtype=np.long)
        d_output_nucs = np.zeros([num_batch, max_edu_seqlen-1], dtype=np.long)
        d_output_rels = np.zeros([num_batch, max_edu_seqlen - 1], dtype=np.long)
        d_masks = np.zeros([num_batch, max_edu_seqlen-1, max_edu_seqlen+1], dtype=np.uint8)

        for batchi, (encoder_inputs, decoder_inputs, pred_splits, pred_nucs, pred_rels) in enumerate(batch):
            for edui, (edu_word_ids, edu_pos_ids) in enumerate(encoder_inputs):
                word_seqlen = len(edu_word_ids)
                e_input_words[batchi][edui][:word_seqlen] = edu_word_ids
                e_input_poses[batchi][edui][:word_seqlen] = edu_pos_ids
                e_masks[batchi][edui][:word_seqlen] = 1

            for di, decoder_input in enumerate(decoder_inputs):
                d_inputs[batchi][di] = decoder_input
                d_masks[batchi][di][decoder_input[0]+1: decoder_input[1]] = 1
            d_outputs[batchi][:len(pred_splits)] = pred_splits
            d_output_nucs[batchi][:len(pred_nucs)] = pred_nucs
            d_output_rels[batchi][:len(pred_rels)] = pred_rels

        # numpy to torch
        e_input_words = torch.from_numpy(e_input_words).long()
        e_input_poses = torch.from_numpy(e_input_poses).long()
        e_masks = torch.from_numpy(e_masks).byte()
        d_inputs = torch.from_numpy(d_inputs).long()
        d_outputs = torch.from_numpy(d_outputs).long()
        d_output_nucs = torch.from_numpy(d_output_nucs).long()
        d_output_rels = torch.from_numpy(d_output_rels).long()
        d_masks = torch.from_numpy(d_masks).byte()

        if use_gpu:
            e_input_words = e_input_words.cuda()
            e_input_poses = e_input_poses.cuda()
            e_masks = e_masks.cuda()
            d_inputs = d_inputs.cuda()
            d_outputs = d_outputs.cuda()
            d_output_nucs = d_output_nucs.cuda()
            d_output_rels = d_output_rels.cuda()
            d_masks = d_masks.cuda()

        yield (e_input_words, e_input_poses, e_masks), (d_inputs, d_masks), (d_outputs, d_output_nucs, d_output_rels)
        offset = offset + batch_size


def parse_and_eval(dataset, model):
    model.eval()
    parser = PartitionPtrParser(model)
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
        strips.append(edus)
    parses = []
    for edus in strips:
        parse = parser.parse(edus)
        parses.append(parse)
    return num_instances, evaluation_trees(parses, golds, treewise_avearge=True)


def gen_report(span_score, nuc_score, ctype_score, ftype_score):
    report = '\n'
    report += '                 precision    recall    f1\n'
    report += '---------------------------------------------\n'
    report += 'span             %5.3f        %5.3f     %5.3f\n' % span_score
    report += 'nuclear          %5.3f        %5.3f     %5.3f\n' % nuc_score
    report += 'ctype            %5.3f        %5.3f     %5.3f\n' % ctype_score
    report += 'ftype            %5.3f        %5.3f     %5.3f\n' % ftype_score
    report += '\n'
    return report


def model_score(scores):
    eval_score = sum(score[2] for score in scores)
    return eval_score


def main(args):
    # set seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load dataset
    cdtb = CDTB(args.data, "TRAIN", "VALIDATE", "TEST", ctb_dir=args.ctb_dir, preprocess=True, cache_dir=args.cache_dir)
    # build vocabulary
    word_vocab, pos_vocab, nuc_label, rel_label = build_vocab(cdtb.train)

    trainset = numericalize(cdtb.train, word_vocab, pos_vocab, nuc_label, rel_label)
    logging.info("num of instances trainset: %d" % len(trainset))
    logging.info("args: %s" % str(args))
    # build model
    model = PartitionPtr(hidden_size=args.hidden_size, dropout=args.dropout,
                         word_vocab=word_vocab, pos_vocab=pos_vocab, nuc_label=nuc_label, rel_label=rel_label,
                         pretrained=args.pretrained, w2v_size=args.w2v_size, w2v_freeze=args.w2v_freeze,
                         pos_size=args.pos_size,
                         split_mlp_size=args.split_mlp_size, nuc_mlp_size=args.nuc_mlp_size,
                         rel_mlp_size=args.rel_mlp_size,
                         use_gpu=args.use_gpu)
    if args.use_gpu:
        model.cuda()
    logging.info("model:\n%s" % str(model))

    # train and evaluate
    niter = 0
    log_splits_loss = 0.
    log_nucs_loss = 0.
    log_rels_loss = 0.
    log_loss = 0.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    writer = SummaryWriter("data/log")
    best_model = None
    best_model_score = 0.
    for nepoch in range(1, args.epoch + 1):
        batch_iter = gen_batch_iter(trainset, args.batch_size, args.use_gpu)
        for nbatch, (e_inputs, d_inputs, grounds) in enumerate(batch_iter, start=1):
            niter += 1
            model.train()
            optimizer.zero_grad()
            splits_loss, nucs_loss, rels_loss = model.loss(e_inputs, d_inputs, grounds)
            loss = args.a_split_loss * splits_loss + args.a_nuclear_loss * nucs_loss + args.a_relation_loss * rels_loss
            loss.backward()
            optimizer.step()
            log_splits_loss += splits_loss.item()
            log_nucs_loss += nucs_loss.item()
            log_rels_loss += rels_loss.item()
            log_loss += loss.item()
            if niter % args.log_every == 0:
                logging.info("[iter %-6d]epoch: %-3d, batch %-5d,"
                             "train splits loss:%.5f, nuclear loss %.5f, relation loss %.5f, loss %.5f" %
                             (niter, nepoch, nbatch, log_splits_loss, log_nucs_loss, log_rels_loss, log_loss))
                writer.add_scalar("train/split_loss", log_splits_loss, niter)
                writer.add_scalar("train/nuclear_loss", log_nucs_loss, niter)
                writer.add_scalar("train/relation_loss", log_rels_loss, niter)
                writer.add_scalar("train/loss", log_loss, niter)
                log_splits_loss = 0.
                log_nucs_loss = 0.
                log_rels_loss = 0.
                log_loss = 0.
            if niter % args.validate_every == 0:
                num_instances, validate_scores = parse_and_eval(cdtb.validate, model)
                logging.info("validation on %d instances" % num_instances)
                logging.info(gen_report(*validate_scores))
                writer.add_scalar("validate/span_f1", validate_scores[0][2], niter)
                writer.add_scalar("validate/nuclear_f1", validate_scores[1][2], niter)
                writer.add_scalar("validate/coarse_relation_f1", validate_scores[2][2], niter)
                writer.add_scalar("validate/fine_relation_f1", validate_scores[3][2], niter)
                new_model_score = model_score(validate_scores)
                if new_model_score > best_model_score:
                    # test on testset with new best model
                    best_model_score = new_model_score
                    best_model = copy.deepcopy(model)
                    logging.info("test on new best model")
                    num_instances, test_scores = parse_and_eval(cdtb.test, best_model)
                    logging.info("test on %d instances" % num_instances)
                    logging.info(gen_report(*test_scores))
                    writer.add_scalar("test/span_f1", test_scores[0][2], niter)
                    writer.add_scalar("test/nuclear_f1", test_scores[1][2], niter)
                    writer.add_scalar("test/coarse_relation_f1", test_scores[2][2], niter)
                    writer.add_scalar("test/fine_relation_f1", test_scores[3][2], niter)
    if best_model:
        # evaluation and save best model
        logging.info("final test result")
        num_instances, test_scores = parse_and_eval(cdtb.test, best_model)
        logging.info("test on %d instances" % num_instances)
        logging.info(gen_report(*test_scores))
        logging.info("save best model to %s" % args.model_save)
        with open(args.model_save, "wb+") as model_fd:
            torch.save(best_model, model_fd)
    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()

    # dataset parameters
    arg_parser.add_argument("data")
    arg_parser.add_argument("--ctb_dir")
    arg_parser.add_argument("--cache_dir")

    # model parameters
    arg_parser.add_argument("-hidden_size", default=512, type=int)
    arg_parser.add_argument("-dropout", default=0.33, type=float)
    w2v_group = arg_parser.add_mutually_exclusive_group(required=True)
    w2v_group.add_argument("-pretrained")
    w2v_group.add_argument("-w2v_size", type=int)
    arg_parser.add_argument("-pos_size", default=30, type=int)
    arg_parser.add_argument("-split_mlp_size", default=64, type=int)
    arg_parser.add_argument("-nuc_mlp_size", default=32, type=int)
    arg_parser.add_argument("-rel_mlp_size", default=128, type=int)
    arg_parser.add_argument("--w2v_freeze", dest="w2v_freeze", action="store_true")
    arg_parser.set_defaults(w2v_freeze=False)

    # train parameters
    arg_parser.add_argument("-epoch", default=20, type=int)
    arg_parser.add_argument("-batch_size", default=64, type=int)
    arg_parser.add_argument("-lr", default=0.001, type=float)
    arg_parser.add_argument("-l2", default=0.0, type=float)
    arg_parser.add_argument("-log_every", default=10, type=int)
    arg_parser.add_argument("-validate_every", default=10, type=int)
    arg_parser.add_argument("-a_split_loss", default=0.3, type=float)
    arg_parser.add_argument("-a_nuclear_loss", default=1.0, type=float)
    arg_parser.add_argument("-a_relation_loss", default=1.0, type=float)
    arg_parser.add_argument("-model_save", default="data/models/treebuilder.partptr.model")
    arg_parser.add_argument("--seed", default=21, type=int)
    arg_parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    arg_parser.set_defaults(use_gpu=False)

    main(arg_parser.parse_args())
