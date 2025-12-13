import os
import docx
import PyPDF2
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

"""
==============================================
DOKUMENTASI ALGORITMA STEMMING - SASTRAWI
==============================================

1. ALGORITMA YANG DIGUNAKAN:
   - Sastrawi menggunakan algoritma Nazief & Adriani
   - Algoritma stemming Bahasa Indonesia yang dikembangkan oleh Bobby Nazief dan Mirna Adriani
   - Berbasis aturan (rule-based) untuk menghilangkan imbuhan (afiks) pada kata

2. CARA KERJA:
   a. Cek Kamus: Periksa apakah kata sudah dalam bentuk kata dasar
   b. Hapus Inflection Suffixes: -lah, -kah, -ku, -mu, -nya
   c. Hapus Derivation Suffixes: -i, -an, -kan
   d. Hapus Derivation Prefix: di-, ke-, se-, te-, be-, me-, pe-
   e. Proses berulang hingga menemukan kata dasar atau tidak ada aturan yang cocok

3. KELEBIHAN:
   - Khusus untuk Bahasa Indonesia
   - Menangani berbagai bentuk imbuhan kompleks
   - Menggunakan kamus kata dasar (±28.000 kata)

4. KETERBATASAN:
   - Tidak semua kata dapat di-stem dengan sempurna
   - Kata serapan asing mungkin tidak dikenali
   - Over-stemming: kata yang seharusnya berbeda menjadi sama
   - Under-stemming: kata berimbuhan tidak ter-stem dengan sempurna
"""

# --- Fungsi membaca file ---
def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

def read_docx(path):
    doc = docx.Document(path)
    return " ".join([para.text for para in doc.paragraphs])

# --- Fungsi tokenizing ---
def tokenize(text):
    text = text.lower()
    return text.split()

# --- Fungsi stemming ---
def create_stemmer():
    """Membuat objek stemmer Sastrawi"""
    factory = StemmerFactory()
    return factory.create_stemmer()

def stem_tokens(tokens, stemmer):
    """
    Melakukan stemming pada list token
    Returns: (stemmed_tokens, unstemmed_words)
    - stemmed_tokens: dict {kata_dasar: frekuensi}
    - unstemmed_words: set kata yang tidak bisa di-stem (tetap sama)
    """
    stemmed_tokens = {}
    unstemmed_words = set()
    
    for token in tokens:
        # Stem kata
        stemmed = stemmer.stem(token)
        
        # Hitung frekuensi kata dasar
        if stemmed in stemmed_tokens:
            stemmed_tokens[stemmed] += 1
        else:
            stemmed_tokens[stemmed] = 1
        
        # Deteksi kata yang tidak ter-stem (kata asli == kata hasil stem)
        if token == stemmed and len(token) > 3:  # Hanya kata > 3 huruf
            # Cek apakah memang kata berimbuhan atau bukan
            # Jika token mengandung pola imbuhan tapi tidak berubah
            has_affix_pattern = any([
                token.startswith(('me', 'di', 'ke', 'se', 'te', 'be', 'pe')),
                token.endswith(('kan', 'an', 'i', 'lah', 'kah', 'nya', 'ku', 'mu'))
            ])
            if has_affix_pattern:
                unstemmed_words.add(token)
    
    return stemmed_tokens, unstemmed_words

# --- Program utama ---
dataset_path = "dataset"

# Inisialisasi stemmer
stemmer = create_stemmer()

query = input("Masukkan query: ")
query_tokens = tokenize(query)

print("\n=== HASIL TOKENIZING QUERY ===")
for token in query_tokens:
    print(token)

# Stem query tokens
query_stemmed = [stemmer.stem(token) for token in query_tokens]
print("\n=== HASIL STEMMING QUERY ===")
for original, stemmed in zip(query_tokens, query_stemmed):
    print(f"{original} -> {stemmed}")

print("\n=== HASIL PENCARIAN ===")

# Loop melalui semua subfolder
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    
    # Skip jika bukan folder
    if not os.path.isdir(folder_path):
        continue
    
    # Cari file di setiap subfolder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip jika bukan file (misalnya sub-subfolder)
        if not os.path.isfile(file_path):
            continue
            
        ext = filename.split(".")[-1].lower()
        
        # Baca file sesuai tipe
        try:
            if ext == "txt":
                content = read_txt(file_path)
            elif ext == "pdf":
                content = read_pdf(file_path)
            elif ext == "docx":
                content = read_docx(file_path)
            else:
                continue
        except Exception as e:
            print(f"Gagal membaca file {filename}: {e}")
            continue

        tokens_file = tokenize(content)

        # Proses stemming pada dokumen
        stemmed_doc, unstemmed_doc = stem_tokens(tokens_file, stemmer)
        
        # Hitung frekuensi kata dasar dari query di dokumen
        freq_stemmed = {stemmer.stem(t): stemmed_doc.get(stemmer.stem(t), 0) 
                        for t in query_tokens}

        # Jika semua frekuensi 0 → skip
        if all(v == 0 for v in freq_stemmed.values()):
            continue

        # Tampilkan hasil
        print("\n" + "="*60)
        print("File:", filename)
        print("Tipe :", ext.upper())
        print("Lokasi:", folder_name)
        print("="*60)
        
        print("\n--- FREKUENSI KATA QUERY (SETELAH STEMMING) ---")
        for original_word in query_tokens:
            stemmed_word = stemmer.stem(original_word)
            count = freq_stemmed[stemmed_word]
            print(f"{original_word} -> {stemmed_word} : {count}")
        
        print(f"\n--- SEMUA KATA DASAR DALAM DOKUMEN (Top 20) ---")
        sorted_stems = sorted(stemmed_doc.items(), key=lambda x: x[1], reverse=True)[:20]
        for stem, count in sorted_stems:
            print(f"{stem}: {count}")
        
        print(f"\nTotal kata unik (setelah stemming): {len(stemmed_doc)}")
        print(f"Total kata dalam dokumen: {len(tokens_file)}")
        
        if unstemmed_doc:
            print(f"\n--- KATA YANG TIDAK DAPAT DI-STEMMING ---")
            print(f"Jumlah: {len(unstemmed_doc)}")
            print("Contoh kata (max 15):", ", ".join(list(unstemmed_doc)[:15]))
        else:
            print("\n--- SEMUA KATA BERHASIL DI-STEMMING ---")