import os
import math
import numpy as np
import PyPDF2
import docx


#=== PEMBACA FILE ===#
def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    t = page.extract_text()
                    if t:
                        text += t + " "
                except:
                    continue
    except:
        return ""
    return text

def read_docx(path):
    doc = docx.Document(path)
    return " ".join([p.text for p in doc.paragraphs])

def load_kamus(path="kamus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def remove_suffix(word):
    for s in ["kan", "an", "i"]:
        if word.endswith(s) and len(word) > len(s) + 2:
            return word[:-len(s)]
    return word

def remove_prefix(word):
    prefixes = [
        "meng","meny","men","mem",
        "me","ber","ter",
        "di","ke","se"
    ]
    for p in prefixes:
        if word.startswith(p) and len(word) > len(p) + 2:
            return word[len(p):]
    return word

def stem_sastrawi(word, kamus):
    # langkah 1: cek kamus
    if word in kamus:
        return word

    # langkah 2: hapus akhiran
    temp = remove_suffix(word)
    if temp in kamus:
        return temp

    # langkah 3: hapus awalan
    temp2 = remove_prefix(temp)
    if temp2 in kamus:
        return temp2

    # gagal stemming
    return word


stopwords = set([
    "yang","dan","atau","di","ke","dari","itu","ini","ada","untuk","pada"
])

def preprocess(text, kamus):
    text = text.lower()
    for c in ",.!?;:-_()[]{}\"'":
        text = text.replace(c, " ")

    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stem_sastrawi(t, kamus) for t in tokens]
    return tokens

def count_terms(tokens):
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return freq

#=== TF-IDF PROCESS ===#
def compute_tf(tokens):
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf

def count_terms(tokens):
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return freq


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
