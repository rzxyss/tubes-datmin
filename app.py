from flask import Flask, render_template, request
import os
from itertools import chain
from utils import read_pdf, read_txt, read_docx, preprocess, compute_tf, compute_idf, vectorize, cosine_sim

app = Flask(__name__)

DATASET_PATH = "dataset"

def get_all_files():
    """Mengumpulkan semua file dari subfolder dataset"""
    all_files = []
    
    # Loop melalui semua folder di dataset
    for folder_name in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        
        # Skip jika bukan folder
        if not os.path.isdir(folder_path):
            continue
        
        # Cari file di setiap subfolder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Skip jika bukan file
            if not os.path.isfile(file_path):
                continue
                
            # Cek ekstensi file
            if "." in filename:
                ext = filename.split(".")[-1].lower()
            else:
                continue  # Skip file tanpa ekstensi
            
            # Filter hanya file yang didukung
            if ext in ["txt", "pdf", "docx"]:
                all_files.append({
                    'full_path': file_path,
                    'filename': filename,
                    'ext': ext
                })
    
    return all_files

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form["query"]
    query_tokens = preprocess(query)

    docs_text = []
    filenames = []
    error_files = []
    warning_messages = []
    
    # Dapatkan semua file dari subfolder
    all_files = get_all_files()
    
    print(f"\n=== MULAI PROSES PENCARIAN ===")
    print(f"Query: {query}")
    print(f"Jumlah file ditemukan: {len(all_files)}")
    
    for file_data in all_files:
        file_name = file_data['filename']
        print(f"\nMemproses: {file_name}")
        
        try:
            text = ""
            
            if file_data['ext'] == "txt":
                text = read_txt(file_data['full_path'])
                print(f"  -> Berhasil membaca TXT")
                
            elif file_data['ext'] == "pdf":
                text = read_pdf(file_data['full_path'])
                
                # Cek jika PDF kosong atau error
                if not text or not text.strip():
                    warning_msg = f"PDF '{file_name}' menghasilkan teks kosong"
                    print(f"  -> Warning: {warning_msg}")
                    warning_messages.append(warning_msg)
                    continue
                    
                print(f"  -> Berhasil membaca PDF ({len(text)} karakter)")
                
            elif file_data['ext'] == "docx":
                text = read_docx(file_data['full_path'])
                print(f"  -> Berhasil membaca DOCX")
                
            else:
                continue
            
            # Validasi panjang teks minimal
            text_stripped = text.strip()
            if len(text_stripped) < 10:
                warning_msg = f"File '{file_name}' terlalu pendek ({len(text_stripped)} karakter)"
                print(f"  -> Warning: {warning_msg}")
                warning_messages.append(warning_msg)
                continue
            
            # Validasi apakah teks hanya berisi whitespace atau karakter tidak berguna
            meaningful_chars = len([c for c in text_stripped if c.isalnum()])
            if meaningful_chars < 5:
                warning_msg = f"File '{file_name}' tidak memiliki konten bermakna"
                print(f"  -> Warning: {warning_msg}")
                warning_messages.append(warning_msg)
                continue
                
            docs_text.append(text)
            filenames.append(file_name)
            print(f"  -> File ditambahkan ke processing")
            
        except Exception as e:
            error_msg = f"Error membaca '{file_name}': {str(e)[:100]}"
            print(f"  -> Error: {error_msg}")
            error_files.append(file_name)
            continue

    print(f"\n=== STATISTIK PROSES ===")
    print(f"File berhasil diproses: {len(docs_text)}")
    print(f"File error: {len(error_files)}")
    print(f"File di-skip: {len(warning_messages)}")
    
    if error_files:
        print(f"File error: {', '.join(error_files[:5])}")
        if len(error_files) > 5:
            print(f"  ... dan {len(error_files) - 5} file lainnya")

    # Jika tidak ada file berhasil dibaca
    if not docs_text:
        error_message = "Tidak ada file yang dapat diproses."
        if error_files:
            error_message += f" {len(error_files)} file bermasalah."
        if warning_messages:
            error_message += f" {len(warning_messages)} file di-skip (kosong/tidak valid)."
        
        print(f"\n=== HASIL AKHIR ===")
        print(error_message)
        
        return render_template("result.html", 
                             results=[], 
                             query=query, 
                             message=error_message)

    # Build vocabulary
    print(f"\n=== MEMBANGUN VOCABULARY ===")
    docs_tokens = [preprocess(text) for text in docs_text]
    vocab = sorted(set(chain.from_iterable(docs_tokens + [query_tokens])))
    print(f"Ukuran vocabulary: {len(vocab)} kata")

    # TF for docs & query
    print(f"=== MENGHITUNG TF ===")
    tf_docs = [compute_tf(tokens) for tokens in docs_tokens]
    tf_query = compute_tf(query_tokens)

    # IDF
    print(f"=== MENGHITUNG IDF ===")
    idf = compute_idf(docs_tokens + [query_tokens])

    # Vectorize
    print(f"=== VECTORIZING ===")
    doc_vectors = [vectorize(tf, idf, vocab) for tf in tf_docs]
    query_vector = vectorize(tf_query, idf, vocab)

    # Calculate similarity
    print(f"=== MENGHITUNG SIMILARITY ===")
    results = []
    for filename, vec in zip(filenames, doc_vectors):
        sim = cosine_sim(query_vector, vec)
        if sim > 0:  # Hanya tampilkan yang memiliki similarity > 0
            results.append((filename, round(sim, 4)))
            print(f"  {filename}: {sim:.4f}")

    # Sort by similarity desc
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n=== HASIL AKHIR ===")
    print(f"Ditemukan {len(results)} dokumen relevan")
    
    # Siapkan pesan tambahan untuk template
    info_message = ""
    if error_files or warning_messages:
        info_message = f"({len(error_files)} file error, {len(warning_messages)} file di-skip)"

    return render_template("result.html", 
                         results=results, 
                         query=query, 
                         info_message=info_message)

if __name__ == "__main__":
    app.run(debug=True)