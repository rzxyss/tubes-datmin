import os
import math
import numpy as np
import PyPDF2
import docx

# PEMBACA FILE
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


# KAMUS & STEMMING
def load_kamus(path="kamus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def remove_suffix(word):
    for suf in ["kan", "an", "i"]:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[:-len(suf)]
    return word

def remove_prefix(word):
    prefixes = [
        "meng", "meny", "men", "mem",
        "me", "ber", "ter",
        "di", "ke", "se"
    ]
    for pre in prefixes:
        if word.startswith(pre) and len(word) > len(pre) + 2:
            return word[len(pre):]
    return word

def stem_sastrawi(word, kamus):
    if word in kamus:
        return word

    temp = remove_suffix(word)
    if temp in kamus:
        return temp

    temp2 = remove_prefix(temp)
    if temp2 in kamus:
        return temp2

    return word


# PREPROCESSING
stopwords = {
    "yang", "dan", "atau", "di", "ke",
    "dari", "itu", "ini", "ada", "untuk", "pada"
}

def preprocess(text, kamus):
    text = text.lower()
    for c in ",.!?;:-_()[]{}\"'":
        text = text.replace(c, " ")

    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [stem_sastrawi(t, kamus) for t in tokens]
    return tokens


# TERM FREQUENCY
def compute_tf(tokens):
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


# PEARSON CORRELATION
def pearson_similarity(tf_q, tf_d, vocab):
    x = []
    y = []

    for term in vocab:
        x.append(tf_q.get(term, 0))
        y.append(tf_d.get(term, 0))

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x) ** 2 for xi in x)
    den_y = sum((yi - mean_y) ** 2 for yi in y)

    if den_x == 0 or den_y == 0:
        return 0

    return num / math.sqrt(den_x * den_y)
