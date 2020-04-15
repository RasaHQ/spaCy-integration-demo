import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
from spacy import displacy

matcher_rules = [
    {"label":"PROGLANG","pattern":[{"LOWER":"python"}]},
    {"label":"PROGLANG","pattern":[{"LOWER":{"REGEX":"(python\\d+\\.?\\d*.?\\d*)"}}]},
    {"label":"PROGLANG","pattern":[{"LOWER":"python"},{"TEXT":{"REGEX":"(\\d+\\.?\\d*.?\\d*)"}}]},
    {"label":"PROGLANG","pattern":[{"LOWER": {"IN": ["node", "nodejs", "js", "javascript"]}}]},
    {"label":"PROGLANG","pattern":[{"LOWER": {"IN": ["node", "nodejs", "js", "javascript"]}}, {"TEXT": {"REGEX": "(\\d+\\.?\\d*.?\\d*)"}}]},
]

if __name__ == "__main__":
    nlp = English()
    ruler = EntityRuler(nlp)
    ruler.add_patterns(matcher_rules)
    nlp.add_pipe(ruler, name="proglang-detector")
    nlp.to_disk("spacy-trained-model")
