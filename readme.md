<img src="square-logo.svg" width=200 height=200 align="right">

# Demo of spaCy in Rasa

This repository contains an example of spaCy in Rasa.

It is maintained by Vincent D. Warmerdam, Research Advocate as [Rasa](https://rasa.com/).

## Start 

To install everything simply run; 

```
make install
```

This repository demonstrates how you can attach a custom spaCy model to Rasa. We're
keeping it simple by only having a NER model that uses a pattern matcher but it should
demonstrate how to hook things up. 

You'll first need to create a new spaCy model and you'll need to save it locally. You
can run the `mkmodel.py` file to do all of this. You can specify which specification 
to load in from the command line. For example, for the basic rules;

```
python mkmodel.py spaCy-rules/basic-rules.jsonl
```

This script will create a folder called `spacy-trained-model` with a simple spaCy model inside of it. Our example will be very lightweight 
for demonstration purposes, in real life this folder might get big. 

You'll notice that the `config.yml` file has a reference to this 
model for entity detection. After training this spaCy will do entity 
recognition for you.

```
rasa train
rasa shell nlu
> i want to talk about python 3.6 # [python 3.6] is now a PROGLANG
> i code with node # [node] is now a PROGLANG
```

Note that we're not training the spaCy model here, we're merely using 
it. All spaCy models must be trained before giving it to Rasa. This is
why spaCy is more commonly used as a featurizer. 

Here's an example of a configuration where we also grab the wordembeddings from spaCy so that we can pass it to the classifier. 

```
language: en

pipeline:
- name: SpacyNLP
  model: "spacy-trained-model"
- name: SpacyTokenizer
- name: SpacyEntityExtractor
  dimensions: ["PROGLANG"]
- name: SpacyFeaturizer
  pooling: mean
- name: CountVectorsFeaturizer
  analyzer: char_wb
  min_ngram: 1
  max_ngram: 4
- name: EmbeddingIntentClassifier
```

## Usecase

When might this be useful? There's a few instances; 

- spaCy has support for multiple languages, so if your assistant needs to speak Dutch, you could use a pretrained spaCy model for that as well as the pretrained vectors 
- spaCy has pretrained models that automatically have support for 
common entities such as people and places 
- spaCy has a large community of specialized pretrained models that you can download, say on legal texts

## Not a Usecase 

The spaCy model can be great if you have a highly customized model
and you'd like to get it into Rasa. It may not be an ideal starting 
point though since spaCy is a tool for general NLP tasks while the 
tools that Rasa offers are in general more specalized for the digital 
assistant usecase. 

## Play 

Feel free to play around with this! Happy hacking!
