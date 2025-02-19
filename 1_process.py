import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
import openai
import tiktoken
from dotenv import load_dotenv
load_dotenv()

# Masukkan API key OpenAI Anda
#openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    """Ekstrak teks dari file PDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_texts_from_pdfs(pdf_folder):
    """Ekstrak teks dari semua PDF di folder."""
    pdf_texts = {}
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            text = extract_text_from_pdf(pdf_path)
            pdf_texts[file_name] = text
    return pdf_texts

def split_text_into_chunks(text, max_tokens=500, model="text-embedding-ada-002"):
    """Pisahkan teks menjadi potongan-potongan kecil berdasarkan jumlah token."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

def create_embeddings(text_chunks, model="text-embedding-ada-002"):
    """Buat embedding untuk setiap potongan teks."""
    response = openai.Embedding.create(
        input=text_chunks,
        model=model
    )
    embeddings = [data['embedding'] for data in response['data']]
    return np.array(embeddings, dtype='float32')

def save_faiss_index(embeddings, index_path):
    """Simpan embedding ke dalam FAISS index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)

# === Pipeline Utama ===

# 1. Folder PDF
pdf_folder = "RPIW/"  # Ganti dengan path folder PDF Anda
index_path = "proses/faiss_index.index"
text_chunks_path = "proses/text_chunks.npy"  # Nama file untuk menyimpan teks chunk

# 2. Ekstrak teks dari semua PDF
print("Mengekstrak teks dari PDF...")
pdf_texts = extract_texts_from_pdfs(pdf_folder)

# 3. Pisahkan teks menjadi potongan kecil dan buat embedding
all_chunks = []
all_embeddings = []
chunk_to_pdf_mapping = []  # Untuk melacak potongan berasal dari file PDF mana

print("Membuat embedding untuk teks...")
for pdf_name, text in pdf_texts.items():
    chunks = split_text_into_chunks(text)
    embeddings = create_embeddings(chunks)

    all_chunks.extend(chunks)
    all_embeddings.append(embeddings)
    chunk_to_pdf_mapping.extend([pdf_name] * len(chunks))

# Gabungkan semua embedding menjadi satu array
all_embeddings = np.vstack(all_embeddings)

# 4. Simpan embedding ke dalam FAISS index
print("Menyimpan embedding ke FAISS index...")
save_faiss_index(all_embeddings, index_path)
print(f"FAISS index berhasil disimpan ke {index_path}")

# 5. Simpan potongan teks (chunk) ke file
np.save(text_chunks_path, np.array(all_chunks, dtype=object))
print(f"Teks chunk berhasil disimpan ke {text_chunks_path}")
