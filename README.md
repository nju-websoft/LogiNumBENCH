# LogiNumBench: Benchmarking Joint Logical and Numerical Reasoning over Natural Language

This repository serves as a generator for the LogiNumBench benchmark. Pre-generated examples can be found at [`data`](./data). Below, you will find a brief introduction to help you generate your own data.

## Quick Start

You can directly use [`demo.py`](./src/demo.py) for generation; you only need to control the parameters passed in. What's more, we provide [`js2disk.py`](./src/js2disk.py) to transform the json file to disk file for datasets library.

To create your custom dataset, first load the template configuration file (you can also add your own), then use the `Theory` class. An instance of this class will automatically handle the sample generation process. Simply pass the desired parameters as follows:

```python
from LogiNumBench import Theory, RandomGene
import json

with open("config.json") as jsf:
    obj = json.load(jsf)
    RandomGene.load_config(obj)
    
while True:
	theory = Theory(id,entityNum, attrNum, factNum, ruleNum, relationNum, depth)
    if theory.assertion is not None:
        break
```

+ `id`: the id for this sample, pass it for generating labeled data in subsequent steps
+ `**Num`: the quantity of ** you want of the data
+ `depth`: the reasoning depth you want of the data

It is essential to note that the depth is intricately linked with the preceding conditions. For instance, a limited quantity of facts and rules may constrain the depth of inference. So, with the parameters you select, there may not exist the reasoning depth you want. In this case, `theory.assertion` will be set to `None` because there is no suitable object for assertion. Since the process is random, you can regenerate until it is suitable. However, please note that the parameters you choose may never be able to generate the reasoning depth you desire.

the parameters we chose are listed below:

| depth | entityNum | attrNum | factNum | ruleNum | relationNum |
| ----- | --------- | ------- | ------- | ------- | ----------- |
| 1     | 4         | 4       | 6       | 6       | 3           |
| 2     | 4         | 4       | 6       | 6       | 3           |
| 3     | 4         | 4       | 6       | 6       | 3           |
| 4     | 5         | 6       | 7       | 7       | 4           |
| 5     | 5         | 6       | 8       | 8       | 5           |
| 6     | 5         | 6       | 8       | 10      | 5           |

## Different Formats

After generating a Theory instance, you can convert it into different forms. 

+ `str(theory)` transforms it into a formal description 
+ `theory.nl()` provides a natural language description of input and target
+ `theory.to_json()` yields a dictionary representation of the components

For example:

```shell
>>> theory = Theory(1, entityNum=4, attrNum=4, factNum=6, ruleNum=6, relationNum=3, depth=3)
>>> str(theory)
"ID: 1\nFact:\n0. argues(Grace, James)\n1. argues(Fiona, Grace)\n2. eats(James, Grace)\n3. is(Fiona, considerate, 1)\n4. eats(Grace, Gary)\n5. argues(Grace, Gary)\nRule:\n0. is(?x, big, 3) -> is(?x, quiet, 2)\n1. hugs(?x, ?y) -> is(?x, quiet, 4y+1)\n2. is(?x, cold, 3) -> is(?x, big, 1)\n3. eats(?x, ?y) -> is(?x, considerate, 7y+2)\n4. is(?x, considerate, 2) -> is(?x, cold, 3)\n5. is(?x, big, 4) -> is(?x, quiet, 2)\nAssertion:\nless(James, cold, 17)\nAnswer:\nTrue\nOne reasoning:\n['Rule4', ['Fact2', 'Rule3', ['Fact4', 'Rule3', ['Default']]]]\nStep reasoning:\nDefault -> int0: Gary is 0 considerate; Fact4 & Rule3 & int0 -> int1: Grace is 7*0+2=2 considerate; Fact2 & Rule3 & int1 -> int2: James is 7*2+2=16 considerate; Rule4 & int2 -> int3: James is 3 cold\n"
>>> theory.nl()
"Fact:\n0. The relationship argues exists between Grace and James.\n1. Fiona is in a state of argues with respect to Grace.\n2. In the context of eats, James and Grace share a connection.\n3. The value 1 is assigned to Fiona for the considerate field.\n4. There is a relation eats between Grace and Gary.\n5. Grace and Gary maintain a connection defined by argues.\nRule:\n0. For x, quiet is represented by the value 2 is a direct result of in x, big is registered as having 3.\n1. The quiet property of x is marked with scaling up the feature quiet of y 4 times and then adding 1 is a foregone conclusion given x and y form a connection of the hugs relationship.\n2. X is defined by 1 in the big field is a natural consequence of within x, the value of cold is 3 being true.\n3. If the eats correlation is present between x and y holds true, it follows that the considerate characteristic of x is taking the characteristic considerate of y and increasing it 7 times with an addition of 2 is inevitable.\n4. If for x, considerate is annotated as 2 holds true, it follows that x is characterized by having 3 in the cold attribute is inevitable.\n5. The value 2 is associated with x under the quiet context can be safely inferred from 4 is attributed to x under big.\nAssertion:\nJames's cold is less than 17.\nStep reasoning:\nDefault -> int0: Gary is 0 considerate; Fact4 & Rule3 & int0 -> int1: Grace is 7*0+2=2 considerate; Fact2 & Rule3 & int1 -> int2: James is 7*2+2=16 considerate; Rule4 & int2 -> int3: James is 3 cold\nAnswer:\nTrue\n"
```



## Insights

In order to better understand the sample generated, even if `theory.assertion` is `None`, you can gain insights into this randomly generated sample through the following method. At the same time, you can also assess the reasonableness of certain parameter.

```shell
>>> theory.states              
[[1, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 90], [0, 0, 0, 9]]
>>> theory.reasoner
[[[('F', 3)], [('F', -1)], [('R', 3), ('S', 0, 0)], [('F', -1)]], [[('F', -1)], [('F', -1)], [('F', -1)], [('F', -1)]], [[('F', -1)], [('F', -1)], [('F', -1)], [('F', 5), ('R', 0), ('S', 3, 3)]], [[('F', -1)], [('F', -1)], [('F', -1)], [('F', 1), ('R', 0), ('S', 0, 3)]]]
```

`theory.states[i][j]`stores the value of attribute j for entity i.

`theory.reasoner[i][j]`stores the source of the value of attribute j for entity i, representing the last step of reasoning that led to obtaining this value. Specifically, there are three forms of the reasoner.

1. `[('F',-1)]` represents that the reasoning source is defaulting to 0.
2. `[('F',i)]` signifies that the value directly comes from the specified fact i.
3. `[('R',i),('S',j,k)]` indicates that the value is derived from the inference of rule j and the value of attribute k for entity j.
4. `[('F',i),('R',j),('S',k,l)]` indicates that the value is derived from the inference of fact i and rule j, and is calculated based on the value of attribute l for entity k.

Now you can easily obtain any information about this sample!

## Add Expression You Want

You can add expression by modify the class `Expr` in [`LogiNumBench.py`](./src/LogiNumBench.py).

1. change the `__init__` to generate your mode
2. complete `compute`,`__str__`, `nl`, `expression_str` functions as the linear operation we have finished 

## Change Assertion Value

If you want to change the assertion value generated, you can just change the code `'num': RandomGene.geneFromInterval(1, 100)` at class `Assertion`.

## Baselines

The [`baselines`](./baselines)directory contains code for training and evaluating various model architectures. Detailed instructions are provided below.

### Requirements

- torch >= 2.0
- transformers
- zhipuai
- openai

### Code

#### Encoder-Only Models

+ [encoder_only_train.py](./baselines/encoder_only/encoder_only_train.py)
+ [encoder_only_eva.py](./baselines/encoder_only/encoder_only_eva.py)

The `encoder_only_train.py` file is used to train the decoder-only models. It trains 12 tasks at once, including D1-6 and LD1-6.

The `encoder_only_eva.py` file is used to evaluate the decoder-only models. The parameters used are set within the file. The config file is used to provide the checkpoint file after training. The format is as follows:

```json
{
    "albert_xlarge": 
    {
        "D1":
        {
            "output":22000,
        },
        "D2":
        {
            "output":14000
        }
    }
}
```

It is recommended to modify the code to adapt to your specific file paths and environment settings. You can use command like the following :

```shell
python encoder_only_train.py gpu=0 ckpt="the path of pre-trained model" dataFile=/"your path"/D{depth}/disk storeFile=/"your path"/D{depth} batchsz="batch size" lr="learning rate"
python encoder_only_eva.py
```

#### Encoder-Decoder Models

+ [encoder_decoder_train.py](./baselines/encoder_decoder/encoder_decoder_train.py)
+ [encoder_decoder_eva.py](./baselines/encoder_decoder/encoder_decoder_eva.py)

Similar to the above, the `encoder_decoder_train.py` file is used to train the encoder-decoder models. It trains 12 tasks at once, including D1-6 and LD1-6.

The `encoder_decoder_eva.py` file is used to evaluate the encoder-decoder models. The parameters used are set within the file. The config file is used to provide the checkpoint file after training. The format is as follows:

```json
{
    "t5_base": {
        "D1": 8000,
        "D2": 8000,
        "D3": 12000
    }
}
```

It is recommended to modify the code to adapt to your specific file paths and environment settings. You can use command like the following :

```shell
python encoder_decoder_train.py gpu=0 ckpt="the path of pre-trained model" dataFile=/"your path"/D{depth}/disk storeFile=/"your path"/D{depth} batchsz="batch size" lr="learning rate"
python encoder_decoder_eva.py
```

#### LLMs

Please configure paths carefully in the files to suit your environment.

+ **ChatGLM3**: To call the API, please refer to the official repository for instructions on running the server.
  + [chatglm3.py](baselines/llms/chatglm3/chatglm3.py)
  + [chatglm3_base.py](baselines/llms/chatglm3/chatglm3_base.py)
+ **Llama2**: To call the API, please refer to the official repository for instructions on running the server.
  + [llama2_chat.py](baselines/llms/llama2/llama2_chat.py)
  + [llama2_text.py](baselines/llms/llama2/llama2_text.py)
+ **Llama3**
  + [llama3_it.py](baselines/llms/llama3/llama3_it.py)
  + [llama3_base.py](baselines/llms/llama3/llama3_base.py)
+ **Llama3.1**
  + [llama3.1_it.py](baselines/llms/llama3.1/llama3.1_it.py)
  + [llama3.1_base.py](baselines/llms/llama3.1/llama3.1_base.py)
+ **Gemma**
  + [gemma_it.py](baselines/llms/gemma/gemma_it.py)
  + [gemma_base.py](baselines/llms/gemma/gemma_base.py)
+ **Qwen1\.5**
  + [qwen_base.py](baselines/llms/qwen/qwen_base.py)
  + [qwen_chat.py](baselines/llms/qwen/qwen_chat.py)
+ **Qwen2**
  + [qwen2_base.py](baselines/llms/qwen2/qwen2_base.py)
  + [qwen2_it.py](baselines/llms/qwen2/qwen2_it.py)

+ **ChatGPT & GPT-4o**
  + [chatgpt_call.py](baselines/llms/chatgpt/chatgpt_call.py)

#### Pretraining

+ [create_mix.py](baselines/pretrain/create_mix.py): The training set we used, with mixed depths, was collected from D1 to D5 for pre-training.
+ [pretrain.py](baselines/pretrain/pretrain.py): Executes the pre-training.
+ [RT_after_pretrain.py](baselines/pretrain/RT_after_pretrain.py): Fine-tunes on RuleTaker after pre-training.
+ [train_on_RT.py](baselines/pretrain/train_on_RT.py): Directly fine-tunes on RuleTaker.
