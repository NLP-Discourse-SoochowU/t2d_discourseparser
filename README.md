## Transition based Bottom-up RST-style Text-level Discourse Parser

<b>-- General Information</b>
```
   This project displays the top-down DRS parser described in our previous paper 
   "Longyin Zhang, Yuqing Xing, Fang Kong, Peifeng Li, and Guodong Zhou. 
   A Top-Down Neural Architecture towards Text-Level Discourse Parsing 
   of Discourse Rhetorical Structure" in ACL2020. For techinical details, 
   please refer to the paper.
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

##### Functions

1. DRS parsing

Run the following command for DRS parsing:
`python3 parser.py source save [-schema schema_name] [-segmenter_name segmenter_name] [--encoding utf-8] [--draw] [--use_gpu]`

- source： 输入为文本文件路径，每行包含一个段落 / the path of input texts where each line refers to a paragraph;
- save: 存储 xml 文件路径，输入文本文件中的每一行会解析为一个段落节点 / path to save the parse trees;
- schema： 解析策略，目前实现了两种解析策略，`topdown` 和 `shiftreduce`，具体实现在 pipeline.py 中，默认使用 `topdown` / different parsing stratedies;
- segmenter_name：EDU 自动分割器名称，目前实现了两个 EDU 分割器, `svm` 和 `gcn`，默认使用 `svm` / different segmentation stratedies;
- encoding： 输入和输出文件的编码，默认 UTF-8 / encoding format, UTF-8 in default;
- draw: 是否在解析完每个段落后可视化篇章树，需要有图形界面和安装了 tkinter / whether draw the tree or not;
- use_gpu：是否使用 GPU 进行解析，默认使用 cpu  / use GPU or not;

如使用该脚本对 `sample.txt` 中的三个段落使用 topdown 策略和 GCN 子句分割器进行分割，并且可视化每个解析结果，使用如下命令：

```shell
python3 parse.py sample.txt sample.xml -schema topdown -segmenter_name gcn --encoding utf-8 --draw
```

2. 性能评价

性能评价使用 `evaluate.py` 脚本完成，包含如下参数：
`python3 evaluate.py data [--ctb_dir ctb_dir] [-schema topdown|shiftreduce] [-segmenter_name svm|gcn] [-use_gold_edu] [--use_gpu]`

- data： CDTB 语料库路径
- ctb_dir： CTB 语料路径，可以给定 data/CTB 则用标准句法，也可以给定 data/CTB_auto 则使用自动句法
- cache_dir: 存放和加载预处理后的语料缓存路径
- schema： 需要评价的解析策略
- segmenter_name：使用的 EDU 分割器
- use_gold_edu： 使用该参数则使用标准的 EDU 分割重新构建关系评价最后的性能
- use_gpu 是否使用 GPU

如需要评价使用标准 EDU 和标准句法时，topdown 解析方法的篇章解析性能：
```shell
python3 evaluate.py data/CDTB --ctb_dir data/CTB --cache_dir data/cache  -schema topdown --use_gold_edu
```

3. 训练模型

为避免 python 依赖关系的问题，模型的所有 python 文件的调用和引用都是从根目录算起。比如我们打算重新训练 segmenter 下面的 GCN 分割器，
我们使用 `segmenter/gcn/train.py` 脚本，那么调用方法为：

```shell
python -m segmenter.gcn.train /data/csun/ChineseDiscourseParser/data/CDTB -model_save data/models/segmenter.gcn.model -pretrained data/pretrained/sgns.renmin.word --ctb_dir data/CTB --cache_dir data/cache --w2v_freeze --use_gpu
```


#### 关键类和接口介绍

1. SegmenterI

分割器的接口，包括将段落切分为句子、将句子切分 EDU 和将段落一步切分为 EDU 三个接口。

2. ParserI

解析器接口，包括将只包含 EDU 的段落组织为篇章树的方法。

3. PipelineI

流水线类，将 SegmenterI 和 ParserI 具体实现组装起来作为完整的篇章结构解析器。

4. EDU、Relation、Sentence、Paragraph、Discourse

分别对应 EDU、关系、句子、段落和篇章的数据结构，可以看成包含列表的列表容器，Paragraph表示篇章树，可以调用 draw 方法可视化。

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
