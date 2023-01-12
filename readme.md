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
the `nlu.yml` file; 

```yml
version: "3.1"
nlu:
- intent: greet
  examples: |
    - hey
    - hello
...

## intent:goodbye
- intent: goodbye
  examples: |
    - bye
    - goodbye
...

## intent:bot_challenge
- intent: bot_challenge
  examples: |
    - are you a bot?
    - are you a human?
...

## intent:talk_code
- intent: talk_code
  examples: |
    - i want to talk about python- How do you do inline delegates in vb.net like python
    - Code to ask yes/no question in javascript
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
  dimensions: ["PERSON", "GPE"]
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
5. In the next steps we generate some features using the `CountVectorsFeaturizer` that will be passed to the `DIETClassifier`. Since we're interested in showing the effect of the `SpacyEntityExtractor` we're only training the algorithm for 1 epoch.

We can train this pipeline and talk to it to see what the effect is. Let's say `Hi I am Vincent from Amsterdam` to this assistant.

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
entities right out of the box. These include people (`PERSON`) as well as
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
    nlp.add_pipe(ruler, name="proglang")

    # define the name of the model as a package
    nlp.meta["name"] = "proglang"
    # save the model to disk
    nlp.to_disk(nlp.meta["name"])
    print(f"spaCy model saved over at {nlp.meta['name']}.")
```

This script will look in the `matcher-rules` folder and it 
will pick up `.jsonl` files that contain rules for the `EntityRuler`.
Once loaded it will construct a spaCy model and save it to disk. After 
saving it to disk, it is a good habbit to make a proper package out 
of it so that your virtualenv is aware. You can do both steps via; 

```
> python mkmodel.py
Will now create model for matcher-rules/proglang.jsonl.
spaCy model saved over at proglang.
```

## Model as a Package

We now have a saved spaCy model on disk. We could already load it 
with spaCy by calling `spacy.load("proglang")` and that means that 
we could also refer to it in out `config.yml`. For local use this 
is fine but for production use-case it would be nicer to properly 
package the spaCy model. Let's run the commands for that. 

```bash
> python -m spacy package proglang . --force
✔ Loaded meta.json from file
proglang/meta.json
✔ Successfully created package 'en_proglang-3.4.1'
en_proglang-3.4.1
```

This command creates a python package folder structure. 

<details>
  <summary><b>See folder structure.</b></summary>
<code><pre>
en_proglang-3.4.1
├── MANIFEST.in
├── en_proglang
│   ├── __init__.py
│   └── en_proglang-3.4.1
│       ├── meta.json
│       ├── ner
│       │   ├── cfg
│       │   ├── model
│       │   └── moves
│       ├── parser
│       │   ├── cfg
│       │   ├── model
│       │   └── moves
│       ├── proglang
│       │   ├── cfg
│       │   └── patterns.jsonl
│       ├── tagger
│       │   ├── cfg
│       │   ├── model
│       │   └── tag_map
│       ├── tokenizer
│       └── vocab
│           ├── key2row
│           ├── lexemes.bin
│           ├── lookups.bin
│           ├── strings.json
│           └── vectors
├── meta.json
└── setup.py

7 directories, 22 files
</pre></code>
</details>

We can tell python to create a tar file that we can pip install. 

```
> cd en_proglang-3.4.1
> python setup.py sdist 
> cd .. 
```

The `en_proglang-3.4.1` now has different contents. 

<details>
  <summary><b>See new folder structure.</b></summary>
<code><pre>
en_proglang-3.4.1
├── MANIFEST.in
├── dist
│   └── en_proglang-3.4.1.tar.gz
├── en_proglang
│   ├── __init__.py
│   ├── en_proglang-3.4.1
│   │   ├── meta.json
│   │   ├── ner
│   │   │   ├── cfg
│   │   │   ├── model
│   │   │   └── moves
│   │   ├── parser
│   │   │   ├── cfg
│   │   │   ├── model
│   │   │   └── moves
│   │   ├── proglang
│   │   │   ├── cfg
│   │   │   └── patterns.jsonl
│   │   ├── tagger
│   │   │   ├── cfg
│   │   │   ├── model
│   │   │   └── tag_map
│   │   ├── tokenizer
│   │   └── vocab
│   │       ├── key2row
│   │       ├── lexemes.bin
│   │       ├── lookups.bin
│   │       ├── strings.json
│   │       └── vectors
│   └── meta.json
├── en_proglang.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── not-zip-safe
│   ├── requires.txt
│   └── top_level.txt
├── meta.json
└── setup.py

9 directories, 30 files
</pre></code>
</details>

But we can now safely install the model as a package.

```
> python -m pip install en_proglang-3.4.1/dist/en_proglang-3.4.1.tar.gz
```

By doing this we can now load the model in two ways from python. 

```
> python 
>>> import spacy 
>>> spacy.load("en_proglang")
<spacy.lang.en.English object at 0x119d0b080>
>>> import en_proglang
>>> en_proglang.load()
<spacy.lang.en.English object at 0x119d593c8>
```

## Configure 

Now that this is packaged up we can refer to it in our `config.yml`. So here's one that refers to the `en_proglang` link we just made.

```yaml
pipeline:
- name: SpacyNLP
  model: "en_proglang"
- name: SpacyTokenizer
- name: SpacyEntityExtractor
  dimensions: ["PERSON", "GPE", "PROGLANG"]
- name: SpacyFeaturizer
  pooling: mean
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: DIETClassifier
  epochs: 1
```

You'll notice that the `config.yml` file has a reference to `en_proglang`.
This is equivalent to running `spacy.load("en_proglang")` and because it is a package we don't need to worry about filepaths. Now this model will be used for entity detection.

With these changes you can rerun the same experiment and now detect programming languages in the text.

```python
> rasa train; rasa shell nlu
"i program using go"              # [go] is now a PROGLANG entity
"i want to talk about python 3.6" # [python 3.6] is now a PROGLANG entity
"i code with node"                # [node] is now a PROGLANG entity
"i live in Amsterdam"             # [Amsterdam] is now a GPE entity
```

This works but maybe we'd like to limit the entities here. We're only
interested in the entities that refer to programming languages and currently
the base model in spaCy is detecting it as an organisation. There's a few 
options here; 

1. we can change the spaCy model and turn off the native models, this would
also make the pipeline faster 
2. we can change the spaCy model and have it use a better (but heavier) english
model like `en_core_web_lg` as a starting point
3. we can also just turn off the base entities from from `config.yml`

Let's do the latter option.

```yaml
pipeline:
- name: SpacyNLP
  model: "en_proglang"
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
"i live in Amsterdam"             # no entity detected
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
