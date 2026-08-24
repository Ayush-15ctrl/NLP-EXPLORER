import spacy
nlp = spacy.load("en_core_web_sm")

def remove_stopwords(text):
    if not text.strip():
        return "", ""

    doc = nlp(text)
    remaining_words = []
    removed_words = []

    for token in doc:
        if token.is_stop:
            removed_words.append(token.text)
        elif not token.is_space:
            remaining_words.append(token.text)

    cleaned_text = " ".join(remaining_words)
    removed_text = " ".join(removed_words)
    return cleaned_text, removed_text





def lemmatize(text):
    if not text.strip():
        return []

    doc = nlp(text)
    result = []

    for token in doc:
        if not token.is_space:
            result.append([
                token.text, token.lemma_, token.pos_
            ])
    return result