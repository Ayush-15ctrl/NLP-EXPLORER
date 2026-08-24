import spacy 
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_sm")

MODELS = {
    "BERT": "bert-base-uncased",
    "GPT-2": "gpt2",
    "T5": "t5-small",
    "DeepSeek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
}

def spacy_tokenization(text):
    if not text.strip(): 
        return []
    doc = nlp(text)
    result = []
    for token in doc: 
        result.append([token.text])
    return result

def modern_tokenization(text, model_name):
    if not text.strip():
        return[]
    model = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokens = tokenizer.tokenize(text)
    result = []
    for i, token in enumerate(tokens):
        result.append([i, token])
    return result
