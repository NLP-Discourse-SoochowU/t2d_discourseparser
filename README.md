## Top-down Text-level DRS Parser

I often fail to access GitHub, so just send emails to zzlynx@outlook.com (Longyin Zhang) if you have any questions.

<b>-- General Information</b>
```
   This project presents the top-down DRS parser described in the paper "Longyin Zhang, 
   Yuqing Xing, Fang Kong, Peifeng Li, and Guodong Zhou. A Top-Down Neural Architecture 
   towards Text-Level Parsing of Discourse Rhetorical Structure (ACL2020)".
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

1. discourse rhetorical structure parsing

Run the following command for DRS parsing:
```shell
python3 parser.py source save [-schema schema_name] [-segmenter_name segmenter_name] [--encoding utf-8] [--draw] [--use_gpu]
```

- source: the path of input texts where each line refers to a paragraph;
- save: path to save the parse trees;
- schema: `shiftreduce` and `topdown` / different parsing strategies;
- segmenter_name: different segmentation strategies;
- encoding: encoding format, UTF-8 in default;
- draw: whether draw the tree or not through the tkinter tool;
- use_gpu：use GPU or not.


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

3. model training

Taking the EDU segmenter for example:
```shell
python -m segmenter.gcn.train data/CDTB -model_save data/models/segmenter.gcn.model -pretrained data/pretrained/sgns.renmin.word --ctb_dir data/CTB --cache_dir data/cache --w2v_freeze --use_gpu
```


#### Key classes and interfaces

1. SegmenterI

The splitter has three interfaces for segmenting paragraphs into sentences, segmenting sentences into EDU, and segmenting paragraphs into EDU in one step, respectively.

2. ParserI

The parser interface transforms Chinese paragraph EDUs into a discourse tree.

3. PipelineI

Pipeline class, which assembles SegmenterI and ParserI as a complete DRS parser.

4. EDU, Relation, Sentence, Paragraph, and Discourse

They correspond to the data structures of EDU, relation, sentence, paragraph, and discourse, respectively, 
which can also be regarded as list containers. Among them, the Paragraph structure represents the discourse 
tree in the Chinese CDTB corpus, and it can be visualized by calling the draw method.


#### Evaluations

In this paper, we report our performance based on the **soft** micro-averaged F1-score as detailed in
the programs. In addition, this project also contains an unpublished **strict** evaluation method where 
the split position is also taken into consideration for more accurate NR prediction performance. 
Specifically, given two adjacent text spans (1, 5) and (6, 8), the upper-layer span is (1, 8), and we directly
take the predicted boundary pair (1, 8) as a correct prediction in the original soft metric, and the 
corresponding Nucl and Rel between the two child nodes are also assigned. Notably, if the predicted split 
position is 3 and the obtained child spans are (1, 3) and (4, 8), then the soft metric still thinks the span
is correct, and the NR relation between (1, 3) and (4, 8) are predicted as that between (1, 5) and (6, 8) for
evaluation. Obviously, the soft metric is far from regorous. In this project, we also display a strict evaluation
method where both the span boundaries and the split points are considered for span prediction evaluation. And 
the performance of this top-down DRS parser under the strict evaluation is **(84.0, 59.0, 54.2, 47.8) (macro-averaged)**.
And one can directly use these evaluation scripts for performance calculation.


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
