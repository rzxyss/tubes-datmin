import os
import docx
import PyPDF2
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

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


# ======================
# PROGRAM UTAMA
# ======================

# Inisialisasi Sastrawi Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Path dataset fixed ke folder "dataset"
dataset_path = "dataset"

print("\n======================================")
print("PENCARIAN KATA DALAM DATASET")
print("======================================")

if not os.path.isdir(dataset_path):
    print(f"ERROR: Folder '{dataset_path}' tidak ditemukan!")
    exit()

# Input kata dari user
search_word = input("\nMasukkan kata yang ingin dicari: ").strip()

if not search_word:
    print("ERROR: Kata tidak boleh kosong!")
    exit()

# Tokenizing dan stemming kata yang dicari
print(f"\n--- Analisis Kata Input: '{search_word}' ---")
search_tokens = tokenize(search_word)
print(f"Token dari input: {search_tokens}")

search_stemmed_tokens, search_stem_mapping, search_unstemmed = stem_tokens(search_tokens, stemmer)
print(f"Hasil stemming: {search_stemmed_tokens}")

if search_unstemmed:
    print(f"Kata yang tidak dapat di-stemming: {sorted(search_unstemmed)}")
else:
    print("Semua kata berhasil di-stemming")

print("\n======================================")
print("MEMPROSES FILE DALAM DATASET...")
print("======================================")

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
        print(f"  {filename}: {count} kemunculan")

# Cek apakah ada file yang mengandung kata tersebut
if not file_frequencies:
    print(f"\nKata '{search_word}' tidak ditemukan di dalam dataset!")
    exit()

# Cari file dengan frekuensi tertinggi
top_file = max(file_frequencies.items(), key=lambda x: x[1]['count'])
top_filename = top_file[0]
top_data = top_file[1]

print("\n======================================")
print("FILE DENGAN KEMUNCULAN TERBANYAK")
print("======================================")
print(f"File: {top_filename}")
print(f"Jumlah kemunculan kata: {top_data['count']}")

# Hitung frekuensi semua token (TOKENIZING)
freq = {}
for t in top_data['tokens']:
    freq[t] = freq.get(t, 0) + 1

# Hitung frekuensi kata dasar (STEMMING)
stem_freq = {}
for st in top_data['stemmed_tokens']:
    stem_freq[st] = stem_freq.get(st, 0) + 1

# Tampilkan hasil tokenizing (hanya 20 kata teratas untuk efisiensi)
print(f"\n--- Kata Hasil Tokenizing (20 teratas): ---")
sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:20]
for word, count in sorted_freq:
    print(f"  {word} = {count}")

# Tampilkan hasil stemming (hanya 20 kata teratas)
print(f"\n--- Kata Dasar Hasil Stemming (20 teratas): ---")
sorted_stem_freq = sorted(stem_freq.items(), key=lambda x: x[1], reverse=True)[:20]
for word, count in sorted_stem_freq:
    print(f"  {word} = {count}")

# Tampilkan kata yang tidak dapat di-stemming (maksimal 50 kata)
if top_data['unstemmed_words']:
    unstemmed_list = sorted(top_data['unstemmed_words'])[:50]
    print(f"\n--- Kata yang tidak dapat di-stemming ({len(top_data['unstemmed_words'])} total, menampilkan {len(unstemmed_list)}): ---")
    for idx, word in enumerate(unstemmed_list, 1):
        print(f"  {idx}. {word}")
else:
    print(f"\nSemua kata berhasil di-stemming")