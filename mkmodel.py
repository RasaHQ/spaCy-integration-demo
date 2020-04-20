import pathlib

import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
from spacy import displacy


if __name__ == "__main__":
    for path in pathlib.Path('spaCy-rules').glob("*.jsonl"):
        # note that we could have also used `en_core_web_md` as a starting point
        # or another pretrained language model, like Dutch `nl_core_news_sm`
        # we're keeping it minimal for now though
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
