## Top-down Text-level DRS Parser

<b>-- General Information</b>
```
   This project presents the top-down DRS parser described in "Longyin Zhang, Yuqing Xing, 
   Fang Kong, Peifeng Li, and Guodong Zhou. A Top-Down Neural Architecture towards Text-Level 
   Parsing of Discourse Rhetorical Structure" in ACL2020.
```

#### Installation
- Python 3.6
- java for the use of Berkeley Parser
- other packages in requirements.txt

#### Project Structure
```
---ChineseDiscourseParser
 |-berkeleyparser Berkeley
 |-data / corpus and models
 |   |-cache / processed data
 |   |-CDTB / the CDTB corpus
 |   |-CTB  / the ChineseTreebank corpus
 |   |-CTB_auto / use Berkeley Parser to parse CTB sentences
 |   |-log / tensorboard log files
 |   |-models / selected model
 |   |-pretrained / word vectors
 |-dataset  / utils for data utilization 
 |   |-cdtb / utils for CDTB processing
 |-pub  
 |-models / related pre-trained models
 |-pyltp_models  / third-party models of pyltp
 |-segmenter  / EDU segmentation
 |   |- gcn / GCN based EDU segmenter
 |   |-rnn  / LSTM-based EDU segmenter
 |   |-svm  / SVM-based EDU segmenter 
 |-structure / tree structures
 |   |-nodes.py  / tree nodes 
 |   |-vocab.py  / vocabolary list
 |-treebuilder  / tree parser
 |   |-partptr  / the top-down DRS parser
 |   |-shiftreduce  / the transition-based DRS parser 
 |-util  / some utils
 |   |-berkeley.py  / functions of Berkeley Parser 
 |   |-eval.py / evaluation methods
 |   |-ltp.py  / PyLTP tools
 |-evaluate.py  / evaluation
 |-interface.py 
 |-parser.py  / parsing with pre-trained parsers
 |-pipeline.py  / a pipelined framework
```

##### Project Functions

1. DRS parsing

Run the following command for DRS parsing:
```shell
python3 parser.py source save [-schema schema_name] [-segmenter_name segmenter_name] [--encoding utf-8] [--draw] [--use_gpu]
```

- source： 输入为文本文件路径，每行包含一个段落 / the path of input texts where each line refers to a paragraph;
- save: 存储 xml 文件路径，输入文本文件中的每一行会解析为一个段落节点 / path to save the parse trees;
- schema： 解析策略，目前实现了两种解析策略，`topdown` 和 `shiftreduce`，具体实现在 pipeline.py 中，默认使用 `topdown` / different parsing stratedies;
- segmenter_name：EDU 自动分割器名称，目前实现了两个 EDU 分割器, `svm` 和 `gcn`，默认使用 `svm` / different segmentation stratedies;
- encoding： 输入和输出文件的编码，默认 UTF-8 / encoding format, UTF-8 in default;
- draw: 是否在解析完每个段落后可视化篇章树，需要有图形界面和安装了 tkinter / whether draw the tree or not;
- use_gpu：是否使用 GPU 进行解析，默认使用 cpu  / use GPU or not;

E.g., parsing the three paragraphs in `sample.txt` in a top-down mode with GCN-based EDU segmentation and drawing them out:

```shell
python3 parse.py sample.txt sample.xml -schema topdown -segmenter_name gcn --encoding utf-8 --draw
```

2. performance evaluation

Run the following command for performance evaluation: 
`python3 evaluate.py data [--ctb_dir ctb_dir] [-schema topdown|shiftreduce] [-segmenter_name svm|gcn] [-use_gold_edu] [--use_gpu]`

- data: the path of the CDTB corpus;
- ctb_dir: the path of the CTB corpus with CTB based on gold standard syntax and CTB_auto based on auto-syntax;
- cache_dir: the path of cached data;
- schema: the evaluation method to use;
- segmenter_name: the EDU segmenter to use; 
- use_gold_edu: whether use Gold EDU or not;
- use_gpu: use GPU or not.

E.g., if use gold EDU and gold syntax for top-down parsing, run:
```shell
python3 evaluate.py data/CDTB --ctb_dir data/CTB --cache_dir data/cache  -schema topdown --use_gold_edu
```

3. model training

Taking EDU segmentation for eaxmple:
```shell
python -m segmenter.gcn.train /data/csun/ChineseDiscourseParser/data/CDTB -model_save data/models/segmenter.gcn.model -pretrained data/pretrained/sgns.renmin.word --ctb_dir data/CTB --cache_dir data/cache --w2v_freeze --use_gpu
```


#### Key classes and interfaces

1. SegmenterI

The splitter has three interfaces for segmenting paragraphs into sentences, segmenting sentences into EDU, and segmenting paragraphs into EDU in one step.

2. ParserI

The parser interface, including a method to organize paragraphs containing only EDU into a chapter tree.

3. PipelineI

Pipeline class, which assembles SegmenterI and ParserI as a complete text structure parser.

4. EDU, Relation, Sentence, Paragraph, and Discourse

They correspond to the data structure of EDU, relation, sentence, paragraph, and chapter respectively. They can be regarded as a list container containing lists. Paragraph represents the chapter tree and can be visualized by calling the draw method.

```

<b>-- License</b>
```
   Copyright (c) 2019, Soochow University NLP research group. All rights reserved.
   Redistribution and use in source and binary forms, with or without modification, are permitted provided that
   the following conditions are met:
   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
      following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
```
