from flask import Flask, render_template, request
import os

from utils import (
    read_txt, read_pdf, read_docx,
    load_kamus, preprocess,
    compute_tf, pearson_similarity
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

    # 1. PREPROCESS QUERY
    query_tokens = preprocess(query, kamus)      # case folding + stopword + stemming
    tf_query = compute_tf(query_tokens)
    query_terms = list(set(query_tokens))        # kata dasar query unik

    # 2. BACA SEMUA DOKUMEN
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
            text = read_pdf(path)
        elif ext == "docx":
            text = read_docx(path)
        else:
            continue

        docs_raw.append(text)
        docs_tokens.append(preprocess(text, kamus))  # preprocessing dokumen
        filenames.append(filename)

    if not docs_tokens:
        return render_template("index.html", error="Dataset kosong")

    # 3. HITUNG SIMILARITY
    vocab = sorted(set(sum(docs_tokens + [query_tokens], [])))
    results = []

    for fname, tokens in zip(filenames, docs_tokens):
        tf_doc = compute_tf(tokens)
        score = pearson_similarity(tf_query, tf_doc, vocab)
        results.append((fname, round(score, 4)))

    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:3]

    # 4. TF QUERY (TOP 3 SAJA)
    query_tf_table = []

    for fname, _ in top_results:
        idx = filenames.index(fname)
        tf_doc = compute_tf(docs_tokens[idx])

        row = {
            "doc": fname,
            "counts": [],
            "total": 0
        }

        for term in query_terms:
            count = tf_doc.get(term, 0)
            row["counts"].append(count)
            row["total"] += count

        query_tf_table.append(row)

    # 5. DOKUMEN TERATAS
    top_file = top_results[0][0]
    top_index = filenames.index(top_file)

    raw_text = docs_raw[top_index][:3000]

    # highlight kata query (berdasarkan kata dasar)
    highlight_text = raw_text
    for q in query_terms:
        highlight_text = highlight_text.replace(
            q, f"<mark>{q}</mark>"
        )

    # 6. PREPROCESSING TAMPILAN
    tokens = docs_raw[top_index].lower().split()
    stemmed_tokens = preprocess(docs_raw[top_index], kamus)

    stem_freq = {}
    for w in stemmed_tokens:
        stem_freq[w] = stem_freq.get(w, 0) + 1

    # 7. RENDER HASIL
    return render_template(
        "result.html",
        query=query,
        results=top_results,
        query_terms=query_terms,
        query_tf_table=query_tf_table,
        top_file=top_file,
        content=highlight_text,
        tokens=tokens[:50],
        cleaned=stemmed_tokens[:50],
        stemmed=stemmed_tokens[:50],
        stem_freq=stem_freq
    )


if __name__ == "__main__":
    app.run(debug=True)
