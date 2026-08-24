import spacy
nlp = spacy.load("en_core_web_sm")

def named_entity_recognition(text):
    if not text.strip():
        return []

    doc = nlp(text)
    result = []
    for entity in doc.ents:
        result.append([
            entity.text,
            entity.label_,
            spacy.explain(entity.label_)
        ])

    return result