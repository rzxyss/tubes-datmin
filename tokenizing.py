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

dataset_path = input("Masukkan path direktori: ")

print("\n======================================")
print("Nama Path:", dataset_path)
print("======================================")

if not os.path.isdir(dataset_path):
    print("ERROR: Path tidak valid!")
    exit()

files = os.listdir(dataset_path)
files.sort()

no = 1

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

    # tokenizing
    tokens = tokenize(content)

    # hitung frekuensi setiap kata (TOKENIZING)
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    # STEMMING - proses stemming pada token
    stemmed_tokens, stem_mapping, unstemmed_words = stem_tokens(tokens, stemmer)
    
    # hitung frekuensi kata dasar (hasil stemming)
    stem_freq = {}
    for st in stemmed_tokens:
        stem_freq[st] = stem_freq.get(st, 0) + 1

    # tampilkan output TOKENIZING
    print(f"\n({no}). {filename}")
    print("Kata Hasil Tokenizing:")

    for word in sorted(freq.keys()):
        print(f"             {word} = {freq[word]}")

    # tampilkan output STEMMING
    print(f"\nKata Dasar Hasil Stemming:")
    for word in sorted(stem_freq.keys()):
        print(f"             {word} = {stem_freq[word]}")
    
    # tampilkan kata yang tidak dapat di-stemming
    if unstemmed_words:
        print(f"\nKata yang tidak dapat di-stemming ({len(unstemmed_words)} kata):")
        unstemmed_sorted = sorted(unstemmed_words)
        for idx, word in enumerate(unstemmed_sorted, 1):
            print(f"             {idx}. {word}")
    else:
        print(f"\nSemua kata berhasil di-stemming")

    no += 1