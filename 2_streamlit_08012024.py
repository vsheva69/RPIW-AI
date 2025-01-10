import os
import streamlit as st
import numpy as np
import faiss
import openai
#from dotenv import load_dotenv
import time
from dotenv import load_dotenv
load_dotenv()

# Masukkan API key OpenAI Anda
openai.api_key = os.getenv("OPENAI_API_KEY")


# === Fungsi Pencarian Gabungan ===
def search_multiple_indices(query, index_dir, model="text-embedding-ada-002", k=5):
    """Cari di beberapa FAISS index dan gabungkan hasilnya."""
    # Buat embedding query
    response = openai.Embedding.create(input=[query], model=model)
    query_embedding = response['data'][0]['embedding']

    all_results = []
    for index_file in sorted(os.listdir(index_dir)):
        if index_file.endswith(".index"):
            # Muat index FAISS
            index_path = os.path.join(index_dir, index_file)
            index = faiss.read_index(index_path)

            # Cari dalam index
            query_vector = np.array([query_embedding], dtype='float32')
            distances, indices = index.search(query_vector, k)

            # Muat potongan teks terkait
            text_chunks_file = index_file.replace(".index", ".npy")
            text_chunks_path = os.path.join(index_dir, text_chunks_file)
            text_chunks = np.load(text_chunks_path, allow_pickle=True).tolist()
            relevant_texts = [text_chunks[i] for i in indices[0]]

            all_results.extend(relevant_texts)

    return all_results

def search_faiss_index(query, index_path, model="text-embedding-ada-002", k=5):
    """Cari embedding yang relevan di FAISS index."""
    # Buat embedding query dengan model yang ditentukan
    response = openai.Embedding.create(
        input=[query],
        model=model
    )
    query_embedding = response['data'][0]['embedding']

    # Muat indeks FAISS
    index = faiss.read_index(index_path)

    # Cari k hasil terdekat
    query_vector = np.array([query_embedding], dtype='float32')
    distances, indices = index.search(query_vector, k)

    return distances, indices

def retrieve_text_from_indices(indices, text_chunks):
    """Ambil teks asli berdasarkan indeks hasil pencarian."""
    return [text_chunks[i] for i in indices[0]]

def ask_gpt4(question, context, model="gpt-4o-mini"):
    """Gunakan GPT-4 untuk menjawab berdasarkan konteks hasil pencarian."""
    messages = [
        {"role": "system", "content": "Anda adalah asisten AI yang membantu menjawab pertanyaan berdasarkan konteks yang diberikan. Anda juga dapat menggunakan akses internet untuk memberikan jawaban yang lebih detail dan faktual"},
        {"role": "user", "content": f"Berikut adalah konteks yang relevan:\n\n{context}\n\nPertanyaan: {question}"}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

def main():
    st.set_page_config(
        page_title="AI RPIW",
        page_icon="🤖",
        layout="wide"
    )

    index_dir = "proses/"

   # Custom CSS to inject
    st.markdown("""
    <style>
    button[kind="secondary"]{
        height: 150px; /* Adjust the height as needed */
        width: 420px; /* Adjust the width as needed */
        border-radius: 15px 50px; 
    }
   @media screen and (max-width: 850px) {
    button[kind="secondary"]{
        height: 50px; /* Adjust the height as needed */
        width: 350px; /* Adjust the width as needed */
        border-radius: 0px; /* Optional: for rounded buttons */
        border-radius: 15px;
                
    }
    }             

    </style>
    """, unsafe_allow_html=True)

    # Main content area for displaying chat messages
    st.title("Chat dengan Dokumen RPIW menggunakan AI🤖")
    st.write("Model: GPT-4o-Mini")


    # Chat input
    # Placeholder for chat messages
    with st.expander('Disclaimer'):
        st.write('''
            Informasi yang disajikan dalam chat ini merupakan hasil analisis AI berdasarkan dokumen asli. Meskipun AI berusaha memberikan informasi yang akurat, kemungkinan masih terdapat ketidaktepatan. Oleh karena itu, hasil analisis ini tidak dapat dijadikan acuan yang sepenuhnya dapat dipertanggungjawabkan kebenarannya.
        ''')

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Selamat Datang. ***Apa yang ingin anda  tanyakan?***"}]
       

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    prompt=''
    satu, dua, tiga,empat = st.columns(4)

    if satu.button("Isu Strategis Provinsi Sumatera Utara",icon=":material/manage_search:", use_container_width=True, type="secondary"):
        prompt= "Apa Isu Strategis perencanaan wilayah di provinsi sumatera utara? "
    if dua.button("Km Jalan Nasional di Provinsi Bangka Belitung", icon=":material/flyover:", use_container_width=True, type="secondary"):
        prompt="Total Panjang Ruas Jalan Nasional di provinsi Bangka Belitung?"
    if tiga.button("Muatan RPIW di Provinsi Aceh?", icon=":material/menu_book:", use_container_width=True, type="secondary"):
        prompt="Apa Saja Muatan RPIW di Provinsi Aceh?"
    if empat.button("Rumah Tidak Layak Huni (RTLH) di Provinsi Lampung", icon=":material/cabin:", use_container_width=True, type="secondary"):
        prompt="Berapa total Rumah Tidak Layak Huni (RTLH) RPIW di Provinsi Lampung?"
        
    chat_input1 = st.chat_input("Ketikan Pertanyaan Anda disini.")  
    if chat_input1:
        prompt = chat_input1

   # prompt = st.chat_input()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt) 
    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Sedang berpikir...."):
                placeholder = st.empty()
	        	# Muat potongan teks (chunk) dari file
                #text_chunks = np.load(text_chunks_path, allow_pickle=True).tolist()
                #distances, indices = search_faiss_index(prompt, index_dir, model="text-embedding-ada-002")
          		# Ambil teks asli yang relevan
                relevant_texts = search_multiple_indices(prompt, index_dir)
                #relevant_texts = retrieve_text_from_indices(indices, text_chunks)
           	 	# Gabungkan hasil teks relevan untuk diberi ke GPT-4
                context = "\n".join(relevant_texts)
                response = ask_gpt4(prompt, context, model="gpt-4o-mini")
                def stream_data():
                    for word in response.split(" "):
                        yield word + " "
                        time.sleep(0.06)
                st.write_stream(stream_data)
        if response is not None:
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
if __name__ == "__main__":
    main()

