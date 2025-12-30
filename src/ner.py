import spacy

nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)
    return [
        (ent.text.lower(), ent.label_)
        for ent in doc.ents
        if ent.label_ in {"ORG", "PRODUCT"}
    ]
