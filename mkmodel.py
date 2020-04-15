import argparse 

import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
from spacy import displacy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filepath', type=str, help='Filepath to .jsonl file containing spaCy rules')
    args = parser.parse_args()

    # note that we could have also used `en_core_web_md` as a starting point
    # or another pretrained language model, like Dutch `nl_core_news_sm`
    # we're keeping it minimal for now though
    nlp = English()

    # create a new rule based NER detector loading in settings from disk
    print(f"Will try to open file at {args.filepath}")
    ruler = EntityRuler(nlp).from_disk(args.filepath)
    print(f"File opened. Will now create model.")
    # add the detector to the model
    nlp.add_pipe(ruler, name="proglang-detector")

    # save the model to disk, this is now also the model name
    # you could load it now via `spacy.load("spacy-trained-model")`
    nlp.to_disk("spacy-trained-model")
    print(f"spaCy model saved in `spacy-trained-model` folder")
