## Transition based Bottom-up RST-style Text-level Discourse Parser

<b>-- General Information</b>
```
   This project displays the top-down DRS parser described in our previous paper 
   "Longyin Zhang, Yuqing Xing, Fang Kong, Peifeng Li, and Guodong Zhou. 
   A Top-Down Neural Architecture towards Text-Level Discourse Parsing 
   of Discourse Rhetorical Structure" in ACL2020. For techinical details, 
   please refer to the paper.
   
   Some people are confused about our evaluation methods, we provide a more detailed description here. 
   For Chinese, we used the original Parseval for evaluation [Morey et al., 2017]. Moreover, like 
   many previous studies, we calculate macro-averaged F1 scores for Chinese trees. 
   For English, we also used the original Parseval for evaluation. At the same time, following previous 
   studies, we report our micro-averaged F1 scores for fair compraison.
```

#### 安装介绍
- Python 3.6
- 安装 java 并放入系统路径（调用 Berkeley Parser 需要）
- requirements.txt 中的 python 依赖，可以用 `pip3 install -r requirements.txt` 安装

#### 工程结构
```
---ChineseDiscourseParser
 |-berkeleyparser Berkeley词法解析器
 |-data  语料库和模型等资源文件
 |   |-cache 预处理语料的缓存
 |   |-CDTB  CDTB 语料
 |   |-CTB   ChineseTreebank 语料
 |   |-CTB_auto 预先使用 Berkeley Parser 重新解析 CTB 句子的词法信息文件
 |   |-log  tensorboard 日志文件
 |   |-models  训练好的模型
 |   |-pretrained  词向量
 |-dataset  语料库操作工具类
 |   |-cdtb  CDTB 语料库工具类，包括加载CDTB CTB、读写、预处理等
 |-pub  发布的信息
 |-models 本任务相关的训练好的模型
 |-pyltp_models  pyltp第三方工具使用的模型
 |-segmenter  EDU 自动分割器
 |   |- gcn 基于 GCN 的自动 EDU 分割器
 |   |-rnn  基于 LSTM 的自动 EDU 分割器
 |   |-svm  基于逗号分类的 EDU 分割器
 |-structure  与篇章树结构相关的数据结构
 |   |-nodes.py  定义了树的各类节点和方法
 |   |-vocab.py  定义了模型用的词表
 |-treebuilder  树的结构生成
 |   |-partptr  基于指针网络的自顶向下的篇章结构解析器
 |   |-shiftreduce  基于转移的中文篇章结构解析器
 |-util   工具类和方法
 |   |-berkeley.py  包装好的 Python 调用 Berkeley Parser 的方法类
 |   |-eval.py  性能评价方法
 |   |-ltp.py  包装好的 PyLTP 工具类
 |-evaluate.py  评价脚本
 |-interface.py  接口和抽象类
 |-parser.py  篇章解析脚本
 |-pipeline.py  定义了不同的解析流水线类
```

##### 功能

1. 篇章解析

篇章解析使用 `parse.py` 脚本完成，`parse.py`的参数如下：
`python3 parser.py source save [-schema schema_name] [-segmenter_name segmenter_name] [--encoding utf-8] [--draw] [--use_gpu]`

- source： 输入为文本文件路径，每行包含一个段落
- save: 存储 xml 文件路径，输入文本文件中的每一行会解析为一个段落节点
- schema： 解析策略，目前实现了两种解析策略，`topdown` 和 `shiftreduce`，具体实现在 pipeline.py 中，默认使用 `topdown`
- segmenter_name：EDU 自动分割器名称，目前实现了两个 EDU 分割器, `svm` 和 `gcn`，默认使用 `svm`
- encoding： 输入和输出文件的编码，默认 UTF-8
- draw: 是否在解析完每个段落后可视化篇章树，需要有图形界面和安装了 tkinter
- use_gpu：是否使用 GPU 进行解析，默认使用 cpu

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
