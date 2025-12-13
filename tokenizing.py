import os
import docx
import PyPDF2
import string

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


# ======================
# PROGRAM UTAMA
# ======================

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

    # hitung frekuensi setiap kata
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    # tampilkan output sesuai format tugas
    print(f"\n({no}). {filename}")
    print("     Mengandung kata:")

    for word in sorted(freq.keys()):
        print(f"             {word} = {freq[word]}")

    no += 1
