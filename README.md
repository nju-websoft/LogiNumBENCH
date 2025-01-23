# LogiNumBench: Synthesizing and Benchmarking Joint Logical and Numerical Reasoning over Natural Language

This repository serves as a synthesizer for the LogiNumBench benchmark. Pre-synthesized examples can be found at [https://doi.org/10.5281/zenodo.14719812](https://doi.org/10.5281/zenodo.14719812). Below, you will find a brief introduction to help you generate your own data.

## Step1: Synthesize a Sample with Template-based Natural Language Description 

You can directly use [`demo.py`](./src/demo.py) for generation; you only need to control the parameters passed in. To create your custom dataset, first load the template and pool configuration file (you can also add your own), then use the `Theory` class. An instance of this class will automatically handle the sample generation process. Simply pass the desired parameters as follows:

```python
from LogiNumBench import Theory, RandomGene
import json

with open("config.json") as jsf:
    obj = json.load(jsf)
    RandomGene.load_config(obj)
    
while True:
	theory = Theory(id, entityNum, attrNum, factNum, ruleNum, relationNum, depth)
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

### Output Formats for Synthesized Theory

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



### Understanding the Synthesized Samples

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

### Extending the theory with custom expressions

You can add expression by modify the class `Expr` in [`LogiNumBench.py`](./src/LogiNumBench.py).

1. change the `__init__` to generate your mode
2. complete `compute`,`__str__`, `nl`, `expression_str` functions as the linear operation we have finished 



## Step2: Refining the Natural Language Description

You can use [fast.py](src/server/fast.py) to deploy a large language model (LLM) and [calllm.py](src/calllm.py) to request optimized natural language descriptions. This step leverages the power of an LLM to ensure that the generated text is both syntactically accurate and semantically coherent, making it suitable for human and language models.

To run the optimization, start by deploying the server:

```shell
uvicorn fast:app --reload --port 8080
```

Then, request the optimized descriptions using the following command:

```shell
python calllm.py
```


## Step3: Filling in Assertion Values Based on Statistical Analysis

We derived the assertion values by analyzing the distribution of values from 24,000 generated samples. By calculating the median and distribution, we selected appropriate parameters for the normal distribution—mean ($\mu$) and standard deviation ($\sigma$). We adjusted the standard deviation so that the range $\mu \pm 3\sigma$ would cover the broadest possible values, while ensuring that no sampled values would fall below zero. Additionally, we set a constraint that all final assertion values must be greater than or equal to 1. This process ensures that assertion values are both statistically grounded and meet the necessary constraints for valid data generation.

To do the above, use [modify_assertion_values.py](src/modify_assertion_values.py).



## The Process of Configuring for Sample Synthesis

We collected an entity pool of 7,944 items from `nltk.corpus.names`, an attribute pool of 1,366 items from `nltk.corpus.treebank`, and a relationship pool of 976 items from `nltk.corpus.wordnet`. These resources are sourced from the Natural Language Toolkit (NLTK), a widely-used library in natural language processing. During the synthesis process, these pools are partitioned across the training, development, and test sets, as shown in the table below.

| **Data split** | **Entity pool** | **Attribute pool** | **Relationship pool** |
| -------------- | --------------- | ------------------ | --------------------- |
| Train          | 6355            | 780                | 1092                  |
| Dev            | 794             | 98                 | 137                   |
| Test           | 795             | 98                 | 137                   |

When entities ($\mathbf{E}$), attributes ($\mathbf{A}$), and relationships ($\mathbf{R}$) are selected from the predefined pools, we ensure that attributes and relationships are chosen without including synonyms. This precaution prevents unintended semantic influence, ensuring the integrity of the study's results.

To obtain the pools for generating samples, we use [getPool.py](src/getPool.py) to gather the data, and [divide_config.py](src/divide_config.py) to split the pools into training, development, and test sets.


## Baselines

The [`baselines`](./baselines) directory contains code for training and evaluating various model architectures. Detailed instructions are provided below.

### Requirements

- torch >= 2.0
- transformers
- openai
- sklearn
- datasets
- pandas
- numpy

### LLMs

To deploy your own LLMs, you can use [server.py](baselines/server.py) and [client.py](baselines/client.py).

For calling external APIs with LLMs, use [do_normal_call.py](baselines/do_normal_call.py).

The instructions and few-shot examples for LLM inputs are located in [instrs](baselines/instrs).

You can analyze the results from LLMs using [analyze_from_text.py](baselines/analyze_from_text.py).

### Encoder-Only Models

To fine-tune encoder-only models for the main experiment, use [encoder-only.py](baselines/encoder-only.py) along with the configuration in [baselines/configs/main-experiment](baselines/configs/main-experiment). For example:

```shell
python encoder-only.py ./configs/main-experiment/albert-base-v2/D1.json
```

To do the cross-distribution experiments, use [cross-eval.py](baselines/cross-eval.py)

To run the data augmentation experiments, use [PretrainingOnLogiNumBENCH.py](baselines/PretrainingOnLogiNumBENCH.py), and for fine-tuning experiments, use [FinetuneOnRuleTaker.py](baselines/FinetuneOnRuleTaker.py). These can be run as direct fine-tuning or fine-tuning after pretraining. The configurations we use for these experiments are located in [baselines/configs/pretraining](baselines/configs/pretraining). Example commands:

```shell
python PretrainingOnLogiNumBENCH.py ./configs/pretraining/bert-base/pretrain.json
python FinetuneOnRuleTaker.py ./configs/pretraining/bert-base/direct.json
```
