from flask import Flask, render_template, request
import os

from utils import (
    read_txt, read_pdf, read_docx,
    load_kamus, preprocess,
    compute_tf, pearson_similarity
)

app = Flask(__name__)
DATASET_PATH = "dataset"

# LOAD KAMUS SEKALI
kamus = load_kamus("kamus.txt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    if not query:
        return render_template("index.html", error="Query tidak boleh kosong")


    # PREPROCESS QUERY

    query_tokens = preprocess(query, kamus)
    tf_query = compute_tf(query_tokens)

    docs_tokens = []
    docs_raw = []
    filenames = []


    # BACA SEMUA DOKUMEN

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

        docs_raw.append(text)
        docs_tokens.append(preprocess(text, kamus))
        filenames.append(filename)


    # PEARSON SIMILARITY (TF)

    vocab = sorted(set(sum(docs_tokens + [query_tokens], [])))
    results = []

    for fname, tokens in zip(filenames, docs_tokens):
        tf_doc = compute_tf(tokens)
        score = pearson_similarity(tf_query, tf_doc, vocab)
        results.append((fname, round(score, 4)))

    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:3]


    # DOKUMEN TERATAS

    top_file = top_results[0][0]
    top_index = filenames.index(top_file)

    raw_text = docs_raw[top_index][:1500]

    # Tahapan preprocessing untuk tampilan
    tokens = docs_raw[top_index].lower().split()
    stemmed_tokens = preprocess(docs_raw[top_index], kamus)

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
