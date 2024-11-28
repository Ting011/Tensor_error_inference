## Initial Experiment
### Paper list

下面两篇工作的style在我看来有一定参考意义：
UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation
(https://arxiv.org/abs/2405.20612)

Reconsidering the Past: Optimizing Hidden States in Language Models
（https://arxiv.org/abs/2112.08653)

顺便一提，涉及量化&模型压缩相关工作本人实在不熟，与个人研究方向有较大偏差。

### Run code
比较需要注意的是实验代码只能在特定Transformer版本下运行
```
Transformer==4.44.2
```

### Simple result
受限于研究领域，我基于整个Transformer的推理流程设计了早期实验：
在llm的推理过程中对各个阶段的hidden state添加错误/偏差，即探索多大的偏差会影响各个层的结果以至于最后的输出结果，目前考虑添加error/bias的位置有(1)before the attention, after embbeding; (2)after attention before full connected layer (3)certain layer of FFN;

实验模型选用的是 Qwen moe 1.5，首先先将推理模块代码从Transformer中提取出来，此处不作详细分析，详见代码；
对源代码做了一些如固定dropout等操作，使得qwen推理过程中能够保证固定输入对应固定输出；
简单debug了一下qwen正常推理过程中各个节点的hidden state格式&值大小，通过直接添加标量作为error value来影响原hidden state；

简单实验结果如下：
(1)before the attention, after embbeding; 
```
```

(2)after attention before full connected layer 
```
```

(3)certain layer of FFN;
```
```
