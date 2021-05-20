# coding: UTF-8
import logging
import argparse
from itertools import chain
from dataset import CDTB
from structure import node_type_filter, EDU, Paragraph, TEXT, Sentence
from pipeline import build_pipeline
from util import eval
from tqdm import tqdm
from util.eval import edu_eval, gen_edu_report

logger = logging.getLogger("evaluation")


def evaluate(args):
    pipeline = build_pipeline(schema=args.schema, segmenter_name=args.segmenter_name, use_gpu=args.use_gpu)
    cdtb = CDTB(args.data, "TRAIN", "VALIDATE", "TEST", ctb_dir=args.ctb_dir, preprocess=True, cache_dir=args.cache_dir)
    golds = list(filter(lambda d: d.root_relation(), chain(*cdtb.test)))
    parses = []

    if args.use_gold_edu:
        logger.info("evaluation with gold edu segmentation")
    else:
        logger.info("evaluation with auto edu segmentation")

    for para in tqdm(golds, desc="parsing", unit=" para"):
        if args.use_gold_edu:
            edus = []
            for edu in para.edus():
                edu_copy = EDU([TEXT(edu.text)])
                setattr(edu_copy, "words", edu.words)
                setattr(edu_copy, "tags", edu.tags)
                edus.append(edu_copy)
        else:
            sentences = []
            for sentence in para.sentences():
                if list(sentence.iterfind(node_type_filter(EDU))):
                    copy_sentence = Sentence([TEXT([sentence.text])])
                    if hasattr(sentence, "words"):
                        setattr(copy_sentence, "words", sentence.words)
                    if hasattr(sentence, "tags"):
                        setattr(copy_sentence, "tags", sentence.tags)
                    setattr(copy_sentence, "parse", cdtb.ctb[sentence.sid])
                    sentences.append(copy_sentence)
            para = pipeline.cut_edu(Paragraph(sentences))
            edus = []
            for edu in para.edus():
                edu_copy = EDU([TEXT(edu.text)])
                setattr(edu_copy, "words", edu.words)
                setattr(edu_copy, "tags", edu.tags)
                edus.append(edu_copy)
        parse = pipeline.parse(Paragraph(edus))
        parses.append(parse)

    # edu score
    scores = edu_eval(golds, parses)
    logger.info("EDU segmentation scores:")
    logger.info(gen_edu_report(scores))

    # parser score
    cdtb_macro_scores = eval.parse_eval(parses, golds, average="macro")
    logger.info("CDTB macro (strict) scores:")
    logger.info(eval.gen_parse_report(*cdtb_macro_scores))

    # nuclear scores
    nuclear_scores = eval.nuclear_eval(parses, golds)
    logger.info("nuclear scores:")
    logger.info(eval.gen_category_report(nuclear_scores))

    # relation scores
    ctype_scores, ftype_scores = eval.relation_eval(parses, golds)
    logger.info("coarse relation scores:")
    logger.info(eval.gen_category_report(ctype_scores))
    logger.info("fine relation scores:")
    logger.info(eval.gen_category_report(ftype_scores))

    # height eval
    height_scores = eval.height_eval(parses, golds)
    logger.info("structure precision by node height:")
    logger.info(eval.gen_height_report(height_scores))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    # dataset parameters
    arg_parser.add_argument("data")
    arg_parser.add_argument("--ctb_dir")
    arg_parser.add_argument("--cache_dir")
    arg_parser.add_argument("-schema", default="topdown")
    arg_parser.add_argument("-segmenter_name", default="gcn")
    arg_parser.add_argument("--use_gold_edu", dest="use_gold_edu", action="store_true")
    arg_parser.set_defaults(use_gold_edu=False)
    arg_parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    arg_parser.set_defaults(use_gpu=False)
    evaluate(arg_parser.parse_args())
