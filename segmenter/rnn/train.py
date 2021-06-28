# coding: UTF-8
import random
from collections import Counter
from structure.vocab import Vocab, Label
from structure.nodes import node_type_filter, EDU, Sentence, Paragraph
from itertools import chain
import numpy as np
from dataset import CDTB
import logging
import argparse
import torch
import torch.optim as optim
from segmenter.rnn.model import RNNSegmenterModel
from segmenter.rnn import RNNSegmenter
from util.eval import edu_eval, gen_edu_report

logger = logging.getLogger("train rnn segmenter")


def build_vocab(dataset):
    word_freq = Counter()
    pos_freq = Counter()
    for paragraph in chain(*dataset):
        for edu in paragraph.edus():
            word_freq.update(edu.words)
            pos_freq.update(edu.tags)
    word_vocab = Vocab("word", word_freq)
    pos_vocab = Vocab("part of speech", pos_freq)
    return word_vocab, pos_vocab


def gen_train_instances(dataset):
    instances = []
    tags = []
    for paragraph in chain(*dataset):
        for sentence in paragraph.sentences():
            edus = list(sentence.iterfind(node_type_filter(EDU)))
            if edus:
                sent_words = []
                sent_poses = []
                sent_tags = []
                for i, edu in enumerate(edus):
                    words = edu.words
                    poses = edu.tags
                    label = ['O'] * (len(words) - 1)
                    label += ['B'] if i < len(edus) - 1 else ['O']
                    sent_words.extend(words)
                    sent_poses.extend(poses)
                    sent_tags.extend(label)
                instances.append((sent_words, sent_poses))
                tags.append(sent_tags)
    return instances, tags


def numericalize(instances, tags, word_vocab, pos_vocab, tag_label):
    trainset = []
    for (words, poses), tags in zip(instances, tags):
        word_ids = [word_vocab[word] for word in words]
        pos_ids = [pos_vocab[pos] for pos in poses]
        tag_ids = [tag_label[tag] for tag in tags]
        trainset.append((word_ids, pos_ids, tag_ids))
    return trainset


def gen_batch_iter(trainset, batch_size, use_gpu=False):
    random_instances = np.random.permutation(trainset)
    num_instances = len(trainset)
    offset = 0
    while offset < num_instances:
        batch = random_instances[offset: min(num_instances, offset + batch_size)]
        num_batch = batch.shape[0]
        lengths = np.zeros(num_batch, dtype=np.int)
        for i, (word_ids, pos_ids, tag_ids) in enumerate(batch):
            lengths[i] = len(word_ids)
        sort_indices = np.argsort(-lengths)
        lengths = lengths[sort_indices]
        batch = batch[sort_indices]
        max_seqlen = lengths.max()
        word_inputs = np.zeros([num_batch, max_seqlen], dtype=np.long)
        pos_inputs = np.zeros([num_batch, max_seqlen], dtype=np.long)
        tag_outputs = np.zeros([num_batch, max_seqlen], dtype=np.long)
        masks = np.zeros([num_batch, max_seqlen], dtype=np.uint8)
        for i, (word_ids, pos_ids, tag_ids) in enumerate(batch):
            seqlen = len(word_ids)
            word_inputs[i][:seqlen] = word_ids
            pos_inputs[i][:seqlen] = pos_ids
            tag_outputs[i][:seqlen] = tag_ids
            masks[i][:seqlen] = 1
        offset = offset + batch_size

        word_inputs = torch.from_numpy(word_inputs).long()
        pos_inputs = torch.from_numpy(pos_inputs).long()
        tag_outputs = torch.from_numpy(tag_outputs).long()
        masks = torch.from_numpy(masks).byte()

        if use_gpu:
            word_inputs = word_inputs.cuda()
            pos_inputs = pos_inputs.cuda()
            tag_outputs = tag_outputs.cuda()
            masks = masks.cuda()
        yield (word_inputs, pos_inputs, masks), tag_outputs


def evaluate(dataset, model):
    model.eval()
    segmenter = RNNSegmenter(model)
    golds = []
    segs = []
    for paragraph in chain(*dataset):
        seged_sents = []
        for sentence in paragraph.sentences():
            # make sure sentence has edus
            if list(sentence.iterfind(node_type_filter(EDU))):
                seged_sents.append(Sentence(segmenter.cut_edu(sentence)))
        if seged_sents:
            segs.append(Paragraph(seged_sents))
            golds.append(paragraph)
    return edu_eval(segs, golds)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main(args):
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("args:" + str(args))
    # load dataset
    cdtb = CDTB(args.data, "TRAIN", "VALIDATE", "TEST", ctb_dir=args.ctb_dir, preprocess=True, cache_dir=args.cache_dir)
    word_vocab, pos_vocab = build_vocab(cdtb.train)
    instances, tags = gen_train_instances(cdtb.train)
    tag_label = Label("tag", Counter(chain(*tags)))
    trainset = numericalize(instances, tags, word_vocab, pos_vocab, tag_label)

    # build model
    model = RNNSegmenterModel(hidden_size=args.hidden_size, dropout=args.dropout, rnn_layers=args.rnn_layers,
                              word_vocab=word_vocab, pos_vocab=pos_vocab, tag_label=tag_label,
                              pos_size=args.pos_size, pretrained=args.pretrained, w2v_freeze=args.w2v_freeze,
                              use_gpu=args.use_gpu)
    if args.use_gpu:
        model.cuda()
    logger.info(model)

    # train
    step = 0
    best_model_f1 = 0
    wait_count = 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)
    for nepoch in range(1, args.epoch+1):
        batch_iter = gen_batch_iter(trainset, args.batch_size, use_gpu=args.use_gpu)
        for nbatch, (inputs, target) in enumerate(batch_iter, start=1):
            step += 1
            model.train()
            optimizer.zero_grad()
            loss = model.loss(inputs, target)
            loss.backward()
            optimizer.step()
            if nbatch > 0 and nbatch % args.log_every == 0:
                logger.info("step %d, patient %d, lr %f, epoch %d, batch %d, train loss %.4f" %
                            (step, wait_count, get_lr(optimizer), nepoch, nbatch, loss.item()))
        # model selection
        score = evaluate(cdtb.validate, model)
        f1 = score[-1]
        scheduler.step(f1, nepoch)
        logger.info("evaluation score:")
        logger.info("\n" + gen_edu_report(score))
        if f1 > best_model_f1:
            wait_count = 0
            best_model_f1 = f1
            logger.info("save new best model to %s" % args.model_save)
            with open(args.model_save, "wb+") as model_fd:
                torch.save(model, model_fd)
            logger.info("test on new best model...")
            test_score = evaluate(cdtb.test, model)
            logger.info("test score:")
            logger.info("\n" + gen_edu_report(test_score))
        else:
            wait_count += 1
            if wait_count > args.patient:
                logger.info("early stopping...")
                break

    with open(args.model_save, "rb") as model_fd:
        best_model = torch.load(model_fd)
    test_score = evaluate(cdtb.test, best_model)
    logger.info("test score on final best model:")
    logger.info("\n" + gen_edu_report(test_score))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    # dataset parameters
    arg_parser.add_argument("data")
    arg_parser.add_argument("--ctb_dir")
    arg_parser.add_argument("--cache_dir")
    arg_parser.add_argument("--seed", default=21, type=int)
    arg_parser.add_argument("-model_save", required=True)

    # model parameter
    arg_parser.add_argument("-hidden_size", default=256, type=int)
    arg_parser.add_argument("-rnn_layers", default=3, type=int)
    arg_parser.add_argument("-dropout", default=0.33, type=float)
    w2v_group = arg_parser.add_mutually_exclusive_group(required=True)
    w2v_group.add_argument("-pretrained")
    w2v_group.add_argument("-w2v_size", type=int)
    arg_parser.add_argument("-pos_size", default=30, type=int)
    arg_parser.add_argument("--w2v_freeze", dest="w2v_freeze", action="store_true")
    arg_parser.set_defaults(w2v_freeze=False)

    # train parameter
    arg_parser.add_argument("-epoch", default=20, type=int)
    arg_parser.add_argument("-lr", default=0.001, type=float)
    arg_parser.add_argument("-l2", default=1e-6, type=float)
    arg_parser.add_argument("-patient", default=4, type=int)
    arg_parser.add_argument("-log_every", default=5, type=int)
    arg_parser.add_argument("-batch_size", default=64, type=int)
    arg_parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    arg_parser.set_defaults(use_gpu=False)
    main(arg_parser.parse_args())
