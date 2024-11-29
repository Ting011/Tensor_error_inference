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

关于结果评估，常用的对于instuct/非instruc model的评估方法往往是在通用数据集上做测试，但由于是早期试验/时间关系/结果不显著，先采用观察结果法并主观验证实验结果吧。

实验模型选用的是 Qwen moe 1.5，首先先将推理模块代码从Transformer中提取出来，此处不作详细分析，详见代码；
对源代码做了一些如固定dropout等操作，使得qwen推理过程中能够保证固定输入对应固定输出；
简单debug了一下qwen正常推理过程中各个节点的hidden state格式&值大小，通过直接添加标量作为error value来影响原hidden state；

简单实验结果如下：
(1)before the attention, after embbeding; 
```
hidden_states = self.model.model.layers[0].input_layernorm(hidden_states) + 0.1/0.01/0.001/0.0001
```
or
```
hidden_states = self.model.model.layers[0].input_layernorm(hidden_states)[0][0] + 0.1/0.01/0.001/0.0001
```

当偏差值到达0.01时，模型部分输出结果有了一定崩坏倾向，但这种现象并不能说明0.01的影响足够大，应该有部分结果仍然正常，受影响结果如下：the answer of the model is [' Hi', 'hi', ',', ' I', ' am', ' a', ' cool', ' person', '.\n', 'I', ' did', ' not', ' notice', ' this', ' one', ' either', '.', ' Maybe', ' no', ' one', ' has', ' noticed', ' this', ' before', '.', ' Everyone', ' noticed', ' this', '.\n', 'Wow', ',', ' wow', '!', ' Wow', ',', ' wow', '!', ' Yes', ' no', ' one', ' else', '.\n']

但上述结果偏差不具备参考性，因为偏差值在0.1 和 1 情况下并不能观察到结果出现明显偏差。
这是可以预见的，因为从transformer模型的推理流程来看，仅仅对attention前的embbeding后hidden state做error增加就相当于给query加了一些微不足道的错误token，这种偏差会被大模型的泛化能力解决；

*special case: 当对hidden state的第一个元素加入大偏差(1)时，会直接导致模型输出EOS触发生成结束，但中间位置加入大偏差就不会有明显影响，有趣，TODO；

(2)after attention before full connected layer 
```
# after self-attention
hidden_states = self.residual + hidden_states + 0.1/0.01/0.001/0.0001
```

在当前部分（attention后，moe前）所添加的偏差值对最终结果有比较显著的影响。主要体现在模型生成EOS终止符的概率提高了很多，这是一个比较主观的结论，但确实能够明显看出生成EOS的概率提高了。
上述现象从总体偏差值为0.01的实验开始出现，这也是可以理解的，因为这个节点的hidden state大小就是0.01级别的，like 0.0139， -0.017
偏差示例如下：
```
# 啥都不吐就直接EOS了
the answer of the model is ['<|endoftext|>']
the answer of the model is ['<|endoftext|>']
finish the token generation
the answer of the model is [' \u200b\u200b', ' I', "'ve", ' decided', ' that', ' there', "'s", ' nothing', ' sex', 'ier', ' than', ' writing', '.', ' Every', ' woman', "'s", ' favorite', ' game', ' is', ' "', 'Where', ' do', ' you', ' want', ' to', ' go', '?"', '<|endoftext|>']
the answer of the model is [' \u200b\u200b', ' I', "'ve", ' decided', ' that', ' there', "'s", ' nothing', ' sex', 'ier', ' than', ' writing', '.', ' Every', ' woman', "'s", ' favorite', ' game', ' is', ' "', 'Where', ' do', ' you', ' want', ' to', ' go', '?"', '<|endoftext|>']
```


(3)certain layer of FFN/MoE or each layer of FFN/MoE;
```
hidden_states = hidden_states + 0.1/0.01/0.001 or hidden state[][] += 0.1/0.01/0.001
# 处理过程略

shared_expert_output = self.after_gate_modules[self.cur_layer]['shared_expert'](hidden_states)
```


在部分case中，出现了明显受影响的现象，如下：
```
the answer of the model is [' —', '—', '\n', 'A', '.', ' —', '—', '\n', 'S', '.', ' o', '\n', 'A', '.', ' —', '—', '\n', 'H', '.', ' e', '\n', 'S', '.', ' o', '\n', 'A', '.', ' —', '—', '——', '\n', 'C', '.', ' li', 'ous', '\n', 'S', '.', ' —', '—', 's', '<|endoftext|>']
```

### On going
1 系统的完成上述实验
2 重新设计评估体系&实验对象，目前在单卡 A6000 上跑实验效率很低，而且7b model在生成质量上很不稳定
3 从参数重要性领域的相关论文中找了一些评估方法与通用数据集，再重复上述实验来做些验证
