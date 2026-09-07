from fastapi import FastAPI
from pydantic import BaseModel

from modules.tokenization import(spacy_tokenization, modern_tokenization)
from modules.preprocessing import(remove_stopwords, lemmatize)
from modules.bow import(bag_of_words)
from modules.ner import(named_entity_recognition)

app = FastAPI(title = "NLP Explorer API", description="REST API for NLP", version="1.0")

class TextRequest(BaseModel):
    text: str
    model: str = "BERT"

@app.get("/")
def home():
    return {"message": "Welcome to NLP Explorer API", "docs": "/docs"}

@app.post("/tokenize")
def tokenize(request: TextRequest):
    result = spacy_tokenization(request.text)
    return {"tokens": result}

@app.post("/modern-tokenize")
def modern_tokenize(request: ModernTokenRequest):
    result = modern_tokenization(request.text, request.model)
    return {"model": request.model, "tokens": result}

@app.post("/stopwords")
def stopwords(request: TextRequest):
    cleaned_text, removed_words = remove_stopwords(request.text)
    return {"cleaned_text": cleaned_text, "removed_stopwords": removed_words}

@app.post("/lemmatize")
def lemmatization(request: TextRequest):
    result = lemmatize(request.text)
    return {"lemmas": result}

@app.post("/bow)
def bow(request: TextRequest):
    result, vocabulary = bag_of_words(request.text)
    return {"word_frequency": result, "vocabulary": vocabulary}

@app.post("/ner")
def ner(request: TextRequest):
    result = named_entity_recognition(request.text)
    return {"entities": result}