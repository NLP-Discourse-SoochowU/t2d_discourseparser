# coding: UTf-8
import logging
import torch
import tqdm
from itertools import chain
from segmenter.rnn import RNNSegmenter
from dataset import CDTB
from structure import node_type_filter, Sentence, Paragraph, EDU
from util.eval import edu_eval, gen_edu_report


logger = logging.getLogger("test rnn segmenter")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with open("data/models/segmenter.rnn.model", "rb") as model_fd:
        model = torch.load(model_fd, map_location='cpu')
        model.use_gpu = False
        model.eval()
    segmenter = RNNSegmenter(model)
    cdtb = CDTB("data/CDTB", "TRAIN", "VALIDATE", "TEST", ctb_dir="data/CTB", preprocess=True, cache_dir="data/cache")

    golds = []
    segs = []
    for paragraph in tqdm.tqdm(chain(*cdtb.test), desc="segmenting"):
        seged_sents = []
        for sentence in paragraph.sentences():
            # make sure sentence has edus
            if list(sentence.iterfind(node_type_filter(EDU))):
                seged_sents.append(Sentence(segmenter.cut_edu(sentence)))
        if seged_sents:
            segs.append(Paragraph(seged_sents))
            golds.append(paragraph)
    scores = edu_eval(segs, golds)
    logger.info(gen_edu_report(scores))
