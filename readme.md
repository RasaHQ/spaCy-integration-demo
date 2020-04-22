<img src="square-logo.svg" width=200 height=200 align="right">

# Demo of spaCy in Rasa

This repository contains an example of how to use spaCy models inside of Rasa.

It is maintained by Vincent D. Warmerdam, Research Advocate as [Rasa](https://rasa.com/).

# spaCy & Rasa

In this guide we're going to show you how you can get a custom spaCy model
working inside of Rasa on your local machine. The document does expect that 
you're already familiar with spaCy and Rasa. If you're not, feel free to check out
the [spaCy online course](https://course.spacy.io/en/) or spaCy [introductory youtube series](https://www.youtube.com/watch?v=WnGPv6HnBok). The getting started guide
for Rasa can be found [here](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/).

If you want to follow along you'll need to clone the repository over [here](https://github.com/RasaHQ/spaCy-integration-demo) and install all the dependencies. 

```
git clone https://github.com/RasaHQ/spaCy-integration-demo
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

#### matcher-rules/proglang.jsonl

This file contains rules that maps patterns to a single entity. 

```json
{"label":"PROGLANG","pattern":[{"LOWER":"golang"}]}
{"label":"PROGLANG","pattern":[{"LOWER":"go", "POS": {"NOT_IN": "VERB"}}]}
{"label":"PROGLANG","pattern":[{"LOWER":"sql"}]}
{"label":"PROGLANG","pattern":[{"LOWER":"python"}]}
{"label":"PROGLANG","pattern":[{"LOWER":{"REGEX":"(python\\d+\\.?\\d*.?\\d*)"}}]}
{"label":"PROGLANG","pattern":[{"LOWER":"python"}, {"TEXT":{"REGEX":"(\\d+\\.?\\d*.?\\d*)"}}]}
{"label":"PROGLANG","pattern":[{"LOWER": {"IN": ["node", "nodejs", "js", "javascript"]}}]}
{"label":"PROGLANG","pattern":[{"LOWER": {"IN": ["node", "nodejs", "js", "javascript"]}}, {"TEXT": {"REGEX": "(\\d+\\.?\\d*.?\\d*)"}}]}
```

Most of the patterns that we're detecting here are based on regex. But the nice 
thing about spaCy matching rules is that we can also use part of speech in these
patterns. That allows us to detect "go" as a programming language, but only if
"go" is not used as a verb. 

## Towards a Model 

To generate the spaCy model using these files locally you can run a script (`mkmodel.py`) that contains the following content;

```python
import pathlib

import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler


if __name__ == "__main__":
    path = pathlib.Path('matcher-rules/proglang.jsonl')
    # note that we could have also used `English()` as a starting point
    # if our matching rules weren't using part of speech 
    nlp = spacy.load("en_core_web_sm")

    # create a new rule based NER detector loading in settings from disk
    ruler = EntityRuler(nlp).from_disk(path)
    print(f"Will now create model for {path}.")

    # add the detector to the model
    nlp.add_pipe(ruler, name="proglang-detector")

    # save the model to disk
    nlp.to_disk("spaCy-custom-model")
    print(f"spaCy model saved.")
```

This script will look in the `spaCy-rules` folder and it 
will pick up `.jsonl` files that contain rules for the `EntityRuler`.
Once loaded it will construct a spaCy model and save it to disk. After 
saving it to disk, it is a good habbit to make a proper package out 
of it so that your virtualenv is aware. You can do both steps via; 

```
> python mkmodel.py
Will now create model for matcher-rules/proglang.jsonl.
spaCy model saved over at custom-proglang-model.
> python -m spacy link custom-proglang-model proglang-model --force
✔ Linking successful
You can now load the model via spacy.load('proglang-model')
```

Once this is on disk we can refer to it in our `config.yml`. So here's 
one that refers to the `proglang-model` link we just made.

```yaml
pipeline:
- name: SpacyNLP
  model: "custom-proglang-model"
- name: SpacyTokenizer
- name: SpacyEntityExtractor
- name: SpacyFeaturizer
  pooling: mean
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 1
```

You'll notice that the `config.yml` file has a reference to `proglang-model`.
This is equivalent to running `spacy.load("proglang-model")` and spaCy has made
a link that ensures it is grabbing the right folder on disk. Now this model will be used for entity detection.

With these changes you can rerun the same experiment and now detect programming
languages in the text.

```python
> rasa train; rasa shell nlu
"i program using go"              # [go] is now a PROGLANG entity
"i want to talk about python 3.6" # [python 3.6] is now a PROGLANG entity
"i code with node"                # [node] is now a PROGLANG entity
"i live in Amsterdam"             # [Amsterdam] is now GPE entity
```

This works but maybe we'd like to limit the entities here, maybe we're only
interested in the entities that refer to programming languages. We can either
change the spaCy model (which would make it faster) but you can also turn 
an entity off from `config.yml`. 

```yaml
pipeline:
- name: SpacyNLP
  model: "proglang-detector"
- name: SpacyTokenizer
- name: SpacyFeaturizer
  pooling: mean
- name: SpacyEntityExtractor
  dimensions: ["PROGLANG"]
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 1
```

We've added a `dimensions` property to `SpacyEntityExtractor` which will 
ensure that we only get entities that we ask for. 


With this configuration, you should now be able to see new behavior.

```python
> rasa train; rasa shell nlu
"i program using go"              # [go] is now a PROGLANG entity
"i want to talk about python 3.6" # [python 3.6] is now a PYTHON entity
"i code with node"                # [node] is now a JAVASCRIPT entity
"i live in amsterdam"             # no entity detected
```

## Usecase

We've only scratched the surface of what is possible with spaCy but hopefully
this guide was able to show you how to you can connect a custom spaCy model to
Rasa.

So you might wonder, when might this be useful? There's a few instances; 

- spaCy has an awesome suite of tools to detect entities and it may just be
that your usecase fits their toolchain really well (like the pattern match
for `go`, when it is not a verb it may be a programming language)
- spaCy has support for multiple languages too, so if your assistant needs to speak Dutch, you could use a pretrained spaCy model for that while still using 
the other tools.
- spaCy has pretrained models that automatically have support for 
common entities such as people and places, meaning you don't need to train 
your own
- spaCy has a large community of specialized pretrained models that you can download, say on legal texts or academic research papers

## Not a Usecase 

That said, you may not need it all the time. 

The spaCy workflow can be great if you have a highly customized model
and you'd like to get it into Rasa. But it may not be an ideal starting 
point though since spaCy is a tool for general NLP tasks while the 
tools that Rasa offers are in general more specalized for the digital 
assistant usecase. 

## Play 

Feel free to play around with this! You can change the empty starting model
in `mkmodel.py` with a beefy pretrained. For the english language `en_core_web_lg` 
is a populaor choice but there's even [multi-lingual models](https://spacy.io/models/xx) to pick from. 

Happy hacking!
