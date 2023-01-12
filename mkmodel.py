import pathlib

import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler


if __name__ == "__main__":
    # note that we could have also used `English()` as a starting point
    # if our matching rules weren't using part of speech 
    nlp = spacy.load("en_core_web_sm")

    rules_file = 'matcher-rules/proglang.jsonl'
    print(f"Will now create model for {rules_file}.")

    # create new entity_ruler pipeline
    ruler = nlp.add_pipe('entity_ruler')

    # load patterns from file
    ruler.from_disk(rules_file)

    # save the model to disk
    nlp.meta["name"] = "proglang"
    nlp.to_disk(nlp.meta["name"])
    print(f"spaCy model saved over at {nlp.meta['name']}.")
