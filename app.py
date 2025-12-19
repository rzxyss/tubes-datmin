from flask import Flask, render_template, request
import os

from utils import (
    read_txt, read_pdf, read_docx,
    load_kamus, preprocess,
    compute_tf, compute_idf,
    vectorize, cosine_sim
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
        return render_template("index.html",
                               error="Query tidak boleh kosong")

    # =====================
    # PREPROCESS QUERY
    # =====================
    query_tokens = preprocess(query, kamus)

    docs_tokens = []
    filenames = []

    # =====================
    # BACA & PREPROCESS DOKUMEN
    # =====================
    for filename in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, filename)

        if not os.path.isfile(path):
            continue

        ext = filename.split(".")[-1].lower()

        if ext == "txt":
            text = read_txt(path)
        elif ext == "pdf":
            text = read_pdf(path)
        elif ext == "docx":
            text = read_docx(path)
        else:
            continue

        docs_tokens.append(preprocess(text, kamus))
        filenames.append(filename)

    # =====================
    # VOCABULARY
    # =====================
    vocab = sorted(set(sum(docs_tokens + [query_tokens], [])))

    # =====================
    # TF & IDF
    # =====================
    tf_docs = [compute_tf(tokens) for tokens in docs_tokens]
    tf_query = compute_tf(query_tokens)

    idf = compute_idf(docs_tokens + [query_tokens])

    # =====================
    # VECTOR SPACE MODEL
    # =====================
    doc_vectors = [
        vectorize(tf, idf, vocab) for tf in tf_docs
    ]
    query_vector = vectorize(tf_query, idf, vocab)

    # =====================
    # COSINE SIMILARITY
    # =====================
    results = []
    for fname, vec in zip(filenames, doc_vectors):
        sim = cosine_sim(query_vector, vec)
        results.append((fname, round(sim, 4)))

    # =====================
    # SORT HASIL
    # =====================
    results.sort(key=lambda x: x[1], reverse=True)

    return render_template(
        "result.html",
        query=query,
        results=results
    )


if __name__ == "__main__":
    app.run(debug=True)
