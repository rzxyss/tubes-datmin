from flask import Flask, render_template, request
import os
import docx
import PyPDF2
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# --- Fungsi membaca file TXT ---
def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# --- Fungsi membaca file PDF ---
def read_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                konten = page.extract_text()
                if konten:
                    text += konten + " "
            except:
                continue
    return text

# --- Fungsi membaca file DOCX ---
def read_docx(path):
    try:
        dok = docx.Document(path)
        return " ".join([p.text for p in dok.paragraphs])
    except:
        return ""

# --- Tokenizing sederhana ---
def tokenize(text):
    text = text.lower()

    # hapus tanda baca
    for p in string.punctuation:
        text = text.replace(p, " ")

    tokens = text.split()
    return tokens

# --- Fungsi Stemming menggunakan Sastrawi ---
def stem_tokens(tokens, stemmer):
    stemmed_tokens = []
    stem_mapping = {}
    unstemmed_words = set()
    
    for token in tokens:
        stemmed = stemmer.stem(token)
        stemmed_tokens.append(stemmed)
        stem_mapping[token] = stemmed
        
        # Identifikasi kata yang tidak berubah (tidak bisa di-stemming)
        if token == stemmed:
            unstemmed_words.add(token)
    
    return stemmed_tokens, stem_mapping, unstemmed_words

# Inisialisasi Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search_word = request.form.get('query', '').strip()
    
    if not search_word:
        return render_template('index.html', error="Kata tidak boleh kosong!")
    
    dataset_path = "dataset"
    
    if not os.path.isdir(dataset_path):
        return render_template('index.html', error=f"Folder '{dataset_path}' tidak ditemukan!")
    
    # Tokenizing dan stemming kata yang dicari
    search_tokens = tokenize(search_word)
    search_stemmed_tokens, search_stem_mapping, search_unstemmed = stem_tokens(search_tokens, stemmer)
    
    # Ambil semua file dari folder dataset
    files = os.listdir(dataset_path)
    files.sort()
    
    # Dictionary untuk menyimpan frekuensi kata di setiap file
    file_frequencies = {}
    
    for filename in files:
        path = os.path.join(dataset_path, filename)
    
        if not os.path.isfile(path):
            continue
    
        ext = filename.split(".")[-1].lower()
    
        # baca file sesuai format
        if ext == "txt":
            content = read_txt(path)
        elif ext == "pdf":
            content = read_pdf(path)
        elif ext == "docx":
            content = read_docx(path)
        else:
            # skip format selain dokumen
            continue
    
        # tokenizing file
        tokens = tokenize(content)
    
        # stemming file
        stemmed_tokens, stem_mapping, unstemmed_words = stem_tokens(tokens, stemmer)
        
        # hitung frekuensi kata yang dicari di file ini
        count = 0
        for stemmed_search in search_stemmed_tokens:
            count += stemmed_tokens.count(stemmed_search)
        
        # simpan informasi file jika kata ditemukan
        if count > 0:
            file_frequencies[filename] = {
                'count': count,
                'tokens': tokens,
                'stemmed_tokens': stemmed_tokens,
                'stem_mapping': stem_mapping,
                'unstemmed_words': unstemmed_words,
                'path': path
            }
    
    # Cek apakah ada file yang mengandung kata tersebut
    if not file_frequencies:
        return render_template('index.html', error=f"Kata '{search_word}' tidak ditemukan di dalam dataset!")
    
    # Urutkan file berdasarkan frekuensi (dari terbanyak ke tersedikit)
    sorted_files = sorted(file_frequencies.items(), key=lambda x: x[1]['count'], reverse=True)
    
    # File teratas (dengan kemunculan terbanyak)
    top_filename = sorted_files[0][0]
    top_data = sorted_files[0][1]
    
    # Hitung frekuensi semua token untuk file teratas (TOKENIZING)
    freq = {}
    for t in top_data['tokens']:
        freq[t] = freq.get(t, 0) + 1
    
    # Hitung frekuensi kata dasar untuk file teratas (STEMMING)
    stem_freq = {}
    for st in top_data['stemmed_tokens']:
        stem_freq[st] = stem_freq.get(st, 0) + 1
    
    # Siapkan data untuk ditampilkan
    # Top 20 kata tokenizing
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Top 20 kata stemming
    sorted_stem_freq = sorted(stem_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Kata yang tidak dapat di-stemming (maksimal 50)
    unstemmed_list = sorted(top_data['unstemmed_words'])[:50]
    
    # Data kata input
    input_data = {
        'word': search_word,
        'tokens': search_tokens,
        'stemmed': search_stemmed_tokens,
        'unstemmed': sorted(search_unstemmed) if search_unstemmed else []
    }
    
    # List semua file dengan frekuensi (untuk ditampilkan)
    all_files = [{'filename': fname, 'count': data['count']} for fname, data in sorted_files]
    
    return render_template('result.html',
                         input_data=input_data,
                         all_files=all_files,
                         top_file=top_filename,
                         top_count=top_data['count'],
                         tokenizing=sorted_freq,
                         stemming=sorted_stem_freq,
                         unstemmed=unstemmed_list,
                         total_unstemmed=len(top_data['unstemmed_words']))

if __name__ == '__main__':
    app.run(debug=True)
