#!/usr/bin/env python3
# coding: UTF-8
import argparse
from structure import Discourse
from pipeline import build_pipeline
import logging
import tqdm


def run(args):
    logger = logging.getLogger("dp")
    doc = Discourse()
    pipeline = build_pipeline(schema=args.schema, segmenter_name=args.segmenter_name, use_gpu=args.use_gpu)
    with open(args.source, "r", encoding=args.encoding) as source_fd:
        for line in tqdm.tqdm(source_fd, desc="parsing %s" % args.source, unit=" para"):
            line = line.strip()
            if line:
                para = pipeline(line)
                if args.draw:
                    para.draw()
                doc.append(para)
    logger.info("save parsing to %s" % args.save)
    doc.to_xml(args.save, encoding=args.encoding)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("source")
    arg_parser.add_argument("save")
    arg_parser.add_argument("-schema", default="topdown")
    arg_parser.add_argument("-segmenter_name", default="svm")
    arg_parser.add_argument("--encoding", default="utf-8")
    arg_parser.add_argument("--draw", dest="draw", action="store_true")
    arg_parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    arg_parser.set_defaults(use_gpu=False)
    arg_parser.set_defaults(draw=False)
    run(arg_parser.parse_args())
