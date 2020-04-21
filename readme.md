<img src="square-logo.svg" width=200 height=200 align="right">

# Demo of spaCy in Rasa

This repository contains an example of how to use spaCy models inside of Rasa.

It is maintained by Vincent D. Warmerdam, Research Advocate as [Rasa](https://rasa.com/).

# spaCy & Rasa

In this guide we're going to show you how you can get a custom spaCy model
working inside of Rasa on your local machine. 

## Install

To install everything you need simply run; 

```
make install
```

## First Config

This repository represents a simple assistant that only
needs to understand four intents. These are described in 
the `nlu.md` file; 

```md
## intent:greet
- hey
- hello
...

## intent:goodbye
- bye
- goodbye
...

## intent:bot_challenge
- are you a bot?
- are you a human?
...

## intent:talk_code
- i want to talk about python- How do you do inline delegates in vb.net like python
- Code to ask yes/no question in javascript
- Executing JavaScript from Flex: Is this javascript function dangerous?
- What does this python error mean? 
...
```

Note that this file only contains intents, we do not have any entities defined here. 

So let's create a `config.yml` file that uses spaCy to detect
entities. 

```yaml
language: en

pipeline:
- name: SpacyNLP
  model: "en_core_web_sm"
- name: SpacyTokenizer
- name: SpacyEntityExtractor
- name: SpacyFeaturizer
  pooling: mean
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 2
- name: DIETClassifier
  epochs: 1

policies:
  - name: MemoizationPolicy
  - name: KerasPolicy
  - name: MappingPolicy
```

Let's note a few things here; 

1. The first step in the pipeline tells us that we're going to use
the `en_core_web_sm` model in spaCy. This is equivalent to calling 
`spacy.load("en_core_web_sm")` which means that you need to make 
sure that it is downloaded beforehand via `python -m spacy download en_core_web_sm`. 
2. Because we're using the spaCy model we now also have to use
the tokenizer from spaCy. We do this is the second pipeline step. 
3. In the third step we're telling spaCy to detect entities on our behalf.
4. In the fourth step we're telling spaCy to also generate the word embeddings
and the pass the mean of these to the next steps.
4. In the next steps we generate some features using the `CountVectorsFeaturizer` that will be passed to the `DIETClassifier`. Since we're interested in showing the effect of the `SpacyEntityExtractor` we're only training the algorithm for 1 epoch.

We can train this pipeline and talk to it to see what the effect is. 

```
> rasa train
> rasa shell nlu
Next message: 
  Hi I am Vincent from Amsterdam. 
```

When you run this, you'll notice in the output that both `Vincent` and `Amsterdam` have been detected as entities. 

```json
...
"entities": [
    {
      "entity": "PERSON",
      "value": "Vincent",
      "start": 9,
      "confidence": null,
      "end": 16,
      "extractor": "SpacyEntityExtractor"
    },
    {
      "entity": "GPE",
      "value": "Amsterdam",
      "start": 22,
      "confidence": null,
      "end": 31,
      "extractor": "SpacyEntityExtractor"
    }
...
```

The standard `en_core_web_sm` spaCy model supports some basic 
entities right out of the box. These both people (`PERSON`) as well as
countries, cities and states (`GPE`). Note that the spaCy model
did not get trained by our `rasa train` command. As far as Rasa is 
concerned spaCy is treated as a pretrained model.

## Customisation 

Let's create our own spaCy model now and add that to the pipeline.
We'll keep it simple by only having a NER model that uses a pattern 
matcher but the general pattern will apply more advanced spaCy models 
as well.

We'll use two files that contain the rules.

#### spaCy-rules/basic-rules.jsonl

This file contains rules that maps patterns to a single entity. 

```json
{"label":"PROGLANG","pattern":[{"LOWER":"golang"}]}
{"label":"PROGLANG","pattern":[{"LOWER":"sql"}]}
{"label":"PROGLANG","pattern":[{"LOWER":"python"}]}
{"label":"PROGLANG","pattern":[{"LOWER":{"REGEX":"(python\\d+\\.?\\d*.?\\d*)"}}]}
{"label":"PROGLANG","pattern":[{"LOWER":"python"}, {"TEXT":{"REGEX":"(\\d+\\.?\\d*.?\\d*)"}}]}
{"label":"PROGLANG","pattern":[{"LOWER": {"IN": ["node", "nodejs", "js", "javascript"]}}]}
{"label":"PROGLANG","pattern":[{"LOWER": {"IN": ["node", "nodejs", "js", "javascript"]}}, {"TEXT": {"REGEX": "(\\d+\\.?\\d*.?\\d*)"}}]}
```

#### spaCy-rules/basic-rules.jsonl

This file contains rules that maps patterns to a multiple entities.

```json
{"label":"GOLANG","pattern":[{"LOWER":"golang"}]}
{"label":"SQL","pattern":[{"LOWER":"sql"}]}
{"label":"PYTHON","pattern":[{"LOWER":"python"}]}
{"label":"PYTHON","pattern":[{"LOWER":{"REGEX":"(python\\d+\\.?\\d*.?\\d*)"}}]}
{"label":"PYTHON","pattern":[{"LOWER":"python"}, {"TEXT":{"REGEX":"(\\d+\\.?\\d*.?\\d*)"}}]}
{"label":"JS","pattern":[{"LOWER": {"IN": ["node", "nodejs", "js", "javascript"]}}]}
{"label":"JS","pattern":[{"LOWER": {"IN": ["node", "nodejs", "js", "javascript"]}}, {"TEXT": {"REGEX": "(\\d+\\.?\\d*.?\\d*)"}}]}
```

## Towards a Model 

To generate the spaCy models using these files locally you can run a script (`mkmodel.py`) that contains the following content;

```python
import pathlib

import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler


if __name__ == "__main__":
    for path in pathlib.Path('spaCy-rules').glob("*.jsonl"):
        # note that we could have also used `en_core_web_sm` as a starting point
        # or another pretrained language model, like Dutch `nl_core_news_sm`
        # we're keeping it lightweight for now though
        nlp = English()

        # create a new rule based NER detector loading in settings from disk
        ruler = EntityRuler(nlp).from_disk(path)
        print(f"Will now create model for {path}.")

        # add the detector to the model
        nlp.add_pipe(ruler, name="proglang-detector")

        # save the model to disk, this is now also the model name
        # you could load it now via `spacy.load("spacy-trained-model")`
        folder = f"spaCy-{str(path.parts[-1]).replace('.jsonl', '')}"
        nlp.to_disk(folder)
        print(f"spaCy model saved in `{folder}` folder")
```

This script will look in the `spaCy-rules` folder and it 
will pick up `.jsonl` files that contain rules for the `EntityRuler`.
Once loaded it will construct a spaCy model and save it to disk. Once
this is on disk we can refer to it in our `config.yml`. So here's 
one that refers to the rules defined in `spaCy-rules/basic-rules.jsonl`.

```yaml
pipeline:
- name: SpacyNLP
  model: "spaCy-basic-rules"
- name: SpacyTokenizer
- name: SpacyEntityExtractor
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 1
```

We've made a few changes now. 

1. You'll notice that the `config.yml` file has a reference to `spaCy-basic-rules`.
This is equivalent to running `spacy.load("spaCy-basic-rules")` because the folder
is on disk. Now this model will be used for entity detection.
2. Since the spaCy model that we've made came from an empty `English()`
model there won't be any word vectors attached. So we've removed the `SpacyFeaturizer`.

With these changes you can rerun the same experiment and now detect programming
languages in the text.

```python
> rasa train; rasa shell nlu
"i want to talk about python 3.6" # [python 3.6] is now a PROGLANG entity
"i code with node"                # [node] is now a PROGLANG entity
```

Let's do this again but with the other spaCy model. 

```yaml
pipeline:
- name: SpacyNLP
  model: "spaCy-more-rules"
- name: SpacyTokenizer
- name: SpacyEntityExtractor
  dimensions: ["GOLANG", "PYTHON", "JAVASCRIPT"]
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 1
```

Here's the differences with the `config.yml` we had before. 

1. We're now referring to the `spaCy-more-rules` model. This was the one
with the matcher rules for for `GOLANG`, `SQL`, `PYTHON` and `JAVASCRIPT` entities. 
2. The entity extractor now has extra settings in it. This configuration will not detect `SQL` but it will detect the other languages. 

With this configuration, you should now be able to see new behavior.

```python
> rasa train; rasa shell nlu
"i want to talk about python 3.6" # [python 3.6] is now a PYTHON entity
"i code with node"                # [node] is now a JAVASCRIPT entity
"this movies deserves a sql"      # no entity detected
"i go for golang"                 # [golang] is now a GOLANG entity
```

## Usecase

We've only scratched the surface of what is possible with spaCy but hopefully
this guide was able to show you how to you can connect a custom spaCy model to
Rasa.

So you might wonder, when might this be useful? There's a few instances; 

- spaCy has support for multiple languages, so if your assistant needs to speak Dutch, you could use a pretrained spaCy model for that.
- spaCy has pretrained models that automatically have support for 
common entities such as people and places 
- spaCy has a large community of specialized pretrained models that you can download, say on legal texts or academic research papers

## Not a Usecase 

That said, you may not need it all the time. 

The spaCy model can be great if you have a highly customized model
and you'd like to get it into Rasa. But it may not be an ideal starting 
point though since spaCy is a tool for general NLP tasks while the 
tools that Rasa offers are in general more specalized for the digital 
assistant usecase. 

## Play 

Feel free to play around with this! Happy hacking!
