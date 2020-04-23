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

    # save the model to disk
    nlp.meta["name"] = "proglang"
    nlp.to_disk(nlp.meta["name"])
    print(f"spaCy model saved over at {nlp.meta['name']}.")
