# coding: UTF-8
import logging
import pickle
import threading
from interface import PipelineI, SegmenterI, ParserI
from segmenter.gcn import GCNSegmenter
from segmenter.svm import SVMSegmenter
from treebuilder.partptr import PartPtrParser
from treebuilder.shiftreduce import ShiftReduceParser
from structure import Paragraph, Sentence
import torch


class BasicPipeline(PipelineI):
    def __init__(self, segmenter, parser):
        super(BasicPipeline, self).__init__()
        self.segmenter = segmenter  # type: SegmenterI
        self.parser = parser  # type: ParserI

    def cut_sent(self, text: str, sid=None):
        return Paragraph(self.segmenter.cut_sent(text, sid=sid))

    def cut_edu(self, para: Paragraph) -> Paragraph:
        edus = []
        for sentence in para.sentences():
            edus.extend(self.segmenter.cut_edu(sentence))
        return Paragraph(edus)

    def parse(self, para: Paragraph) -> Paragraph:
        return self.parser.parse(para)

    def full_parse(self, text: str):
        para = self.cut_sent(text)
        para = self.cut_edu(para)
        para = self.parse(para)
        return para


class ShiftReducePipeline(BasicPipeline):
    def __init__(self, segmenter_name="gcn", use_gpu=False):
        if use_gpu and (not torch.cuda.is_available()):
            raise Warning("cuda is not available, set use_gpu to False")
        if segmenter_name == "svm":
            with open("pub/models/segmenter.svm.model", "rb") as segmenter_fd:
                segmenter_model = pickle.load(segmenter_fd)
            segmenter = SVMSegmenter(segmenter_model)
        elif segmenter_name == "gcn":
            with open("pub/models/segmenter.gcn.model", "rb") as segmenter_fd:
                segmenter_model = torch.load(segmenter_fd, map_location="cpu")
                segmenter_model.use_gpu = False
                if use_gpu:
                    segmenter_model.cuda()
                    segmenter_model.use_gpu = True
                segmenter_model.eval()
                segmenter = GCNSegmenter(segmenter_model)
        with open("pub/models/treebuilder.shiftreduce.model", "rb") as parser_fd:
            parser_model = torch.load(parser_fd, map_location="cpu")
            parser_model.use_gpu = False
            if use_gpu:
                parser_model.cuda()
                parser_model.use_gpu = True
            parser_model.eval()
            parser = ShiftReduceParser(parser_model)
        super(ShiftReducePipeline, self).__init__(segmenter, parser)


class TopDownPipeline(BasicPipeline):
    def __init__(self, segmenter_name="gcn", use_gpu=False):
        if use_gpu and (not torch.cuda.is_available()):
            raise Warning("cuda is not available, set use_gpu to False")
        if segmenter_name == "svm":
            with open("pub/models/segmenter.svm.model", "rb") as segmenter_fd:
                segmenter_model = pickle.load(segmenter_fd)
            segmenter = SVMSegmenter(segmenter_model)
        elif segmenter_name == "gcn":
            with open("pub/models/segmenter.gcn.model", "rb") as segmenter_fd:
                segmenter_model = torch.load(segmenter_fd, map_location="cpu")
                segmenter_model.use_gpu = False
                if use_gpu:
                    segmenter_model.cuda()
                    segmenter_model.use_gpu = True
                segmenter_model.eval()
                segmenter = GCNSegmenter(segmenter_model)
        else:
            raise NotImplemented("no segmenter found for name \"%s\"" % segmenter_name)
        with open("pub/models/treebuilder.partptr.model", "rb") as parser_fd:
            parser_model = torch.load(parser_fd, map_location="cpu")
            parser_model.use_gpu = False
            if use_gpu:
                parser_model.cuda()
                parser_model.use_gpu = True
            parser_model.eval()
            parser = PartPtrParser(parser_model)
        super(TopDownPipeline, self).__init__(segmenter, parser)


def build_pipeline(schema="topdown", segmenter_name="gcn", use_gpu=False):
    logging.info("parsing thread %s build pipeline with %s schema and %s segmenter" %
                 (threading.current_thread().name, schema, segmenter_name))

    if schema == "topdown":
        return TopDownPipeline(segmenter_name=segmenter_name, use_gpu=use_gpu)
    elif schema == "shiftreduce":
        return ShiftReducePipeline(segmenter_name=segmenter_name, use_gpu=use_gpu)
    else:
        raise NotImplemented("no schema found for \"%s\"" % schema)
