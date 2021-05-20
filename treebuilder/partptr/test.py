# coding: UTF-8
import torch
import logging
from itertools import chain

from tqdm import tqdm

from treebuilder.partptr.parser import PartitionPtrParser
from structure.nodes import EDU, TEXT
from dataset import CDTB
from util import eval


def main():
    logging.basicConfig(level=logging.INFO)
    with open("data/models/treebuilder.partptr.model", "rb") as model_fd:
        model = torch.load(model_fd, map_location="cpu")
        model.eval()
        model.use_gpu = False
    parser = PartitionPtrParser(model)
    cdtb = CDTB("data/CDTB", "TRAIN", "VALIDATE", "TEST", ctb_dir="data/CTB", preprocess=True, cache_dir="data/cache")
    golds = list(filter(lambda d: d.root_relation(), chain(*cdtb.test)))

    import parse
    pipeline = parse.build_pipeline()

    strips = []
    for paragraph in golds:
        edus = []
        for edu in paragraph.edus():
            # edu_copy = EDU([TEXT(edu.text)])
            # setattr(edu_copy, "words", edu.words)
            # setattr(edu_copy, "tags", edu.tags)
            edus.append(edu.text)
        strips.append("".join(edus))
        # print(strips[-1])
    parses = []
    parse_sessions = []
    for edus in tqdm(strips):
        # parse, session = parser.parse(edus, ret_session=True)
        parse = pipeline(edus)
        parses.append(parse)
        # parse_sessions.append(session)

    # macro cdtb scores
    cdtb_macro_scores = eval.parse_eval(parses, golds, average="macro")
    logging.info("CDTB macro (strict) scores:")
    logging.info(eval.gen_parse_report(*cdtb_macro_scores))
    # micro cdtb scores
    cdtb_micro_scores = eval.parse_eval(parses, golds, average="micro")
    logging.info("CDTB micro (strict) scores:")
    logging.info(eval.gen_parse_report(*cdtb_micro_scores))

    # micro rst scores
    rst_scores = eval.rst_parse_eval(parses, golds)
    logging.info("RST styled scores:")
    logging.info(eval.gen_parse_report(*rst_scores))

    # nuclear scores
    nuclear_scores = eval.nuclear_eval(parses, golds)
    logging.info("nuclear scores:")
    logging.info(eval.gen_category_report(nuclear_scores))

    # relation scores
    ctype_scores, ftype_scores = eval.relation_eval(parses, golds)
    logging.info("coarse relation scores:")
    logging.info(eval.gen_category_report(ctype_scores))
    logging.info("fine relation scores:")
    logging.info(eval.gen_category_report(ftype_scores))

    # draw gold and parse tree along with decision hotmap
    for gold, parse, session in zip(golds, parses, parse_sessions):
        gold.draw()
        session.draw_decision_hotmap()
        parse.draw()


if __name__ == '__main__':
    main()
