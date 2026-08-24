from sklearn.feature_extraction.text import CountVectorizer

def bag_of_words(text):
    if not text.strip():
        return [], ""

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    counts = matrix.toarray()[0]
    result = []

    for word, count in zip(words, counts):
        result.append([
            word,
            int(count)
        ])

    vocabulary = ", ".join(words)
    return result, vocabulary
