from flask import Flask, render_template, request
import os

from utils import (
    read_txt, read_pdf, read_docx,
    load_kamus, preprocess,
    compute_tf, compute_idf,
    vectorize, cosine_sim,
    count_terms
)

app = Flask(__name__)

DATASET_PATH = "dataset"

# =====================
# LOAD KAMUS (SATU KALI)
# =====================
kamus = load_kamus("kamus.txt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    if not query:
        return render_template("index.html", error="Query tidak boleh kosong")

    # === PREPROCESS QUERY ===
    query_tokens = preprocess(query, kamus)

    docs_tokens = []
    docs_raw = []
    filenames = []

    for filename in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, filename)
        if not os.path.isfile(path):
            continue

        ext = filename.split(".")[-1].lower()
        if ext == "txt":
            text = read_txt(path)
        elif ext == "pdf":
            try:
                text = read_pdf(path)
            except:
                continue
        elif ext == "docx":
            text = read_docx(path)
        else:
            continue

        docs_raw.append(text)
        docs_tokens.append(preprocess(text, kamus))
        filenames.append(filename)

    # === VSM PROCESS ===
    vocab = sorted(set(sum(docs_tokens + [query_tokens], [])))

    tf_docs = [compute_tf(tokens) for tokens in docs_tokens]
    tf_query = compute_tf(query_tokens)

    idf = compute_idf(docs_tokens)

    doc_vectors = [vectorize(tf, idf, vocab) for tf in tf_docs]
    query_vector = vectorize(tf_query, idf, vocab)

    results = []
    for fname, vec in zip(filenames, doc_vectors):
        sim = cosine_sim(query_vector, vec)
        results.append((fname, round(sim, 4)))

    results.sort(key=lambda x: x[1], reverse=True)

    # === AMBIL TOP 3 ===
    top_results = results[:3]

    # === DOKUMEN TERATAS ===
    top_file = top_results[0][0]
    top_index = filenames.index(top_file)

    raw_text = docs_raw[top_index][:1500]  # cuplikan isi

 # =============================
# PREPROCESSING DOKUMEN TERATAS
# =============================

    # Tokenizing (sebelum filtering & stemming)
    tokens = docs_raw[top_index].lower().split()

    # Filtering + Stemming (algoritma sastrawi manual)
    stemmed_tokens = preprocess(docs_raw[top_index], kamus)

    # Hitung frekuensi kata dasar
    stem_freq = {}
    for w in stemmed_tokens:
        stem_freq[w] = stem_freq.get(w, 0) + 1


    return render_template(
        "result.html",
        query=query,
        results=top_results,
        top_file=top_file,
        content=raw_text,
        tokens=tokens[:50],
        cleaned=stemmed_tokens[:50],
        stemmed=stemmed_tokens[:50],
        stem_freq=stem_freq
    )


if __name__ == "__main__":
    app.run(debug=True)
