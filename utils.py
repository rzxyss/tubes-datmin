import os
import math
import numpy as np
import PyPDF2
import docx
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#=== PEMBACA FILE ===#
def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + " "
    return text

def read_docx(path):
    doc = docx.Document(path)
    return " ".join([p.text for p in doc.paragraphs])


#=== PREPROCESSING (TOKENIZING + STOPWORD + STEMMING) ===#
factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopwords = set([
    "yang","dan","atau","di","ke","dari","itu","ini","ada","untuk","pada"
])

def preprocess(text):
    text = text.lower()
    for c in ",.!?;:-_()[]{}\"'":
        text = text.replace(c, " ")
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


#=== TF-IDF PROCESS ===#
def compute_tf(tokens):
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf

def compute_idf(all_docs):
    idf = {}
    total_docs = len(all_docs)

    for doc in all_docs:
        for term in set(doc):
            idf[term] = idf.get(term, 0) + 1

    for term, df in idf.items():
        idf[term] = math.log(total_docs / (1 + df))
    return idf

def vectorize(tf, idf, vocab):
    vec = []
    for term in vocab:
        vec.append(tf.get(term, 0) * idf.get(term, 0))
    return np.array(vec)


#=== COSINE SIMILARITY ===#
def cosine_sim(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
