import os
import streamlit as st
import numpy as np
import faiss
import openai
import pydeck
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

# Masukkan API key OpenAI Anda
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Fungsi Pencarian Gabungan ===
def search_multiple_indices(query, index_dir, model="text-embedding-ada-002", k=10):
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

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    return json.loads(response)

def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)
        st.write("***Tabel:***")
        st.write(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.line_chart(df)
        st.write("***Tabel:***")
        st.write(df)
        
    if "map" in response_dict:
        data = response_dict["map"]
        df= pd.DataFrame(data)
        df["size"] = 10000
        lat_first = df["Lat"].iloc[1]
        lon_last = df["Long"].iloc[0]
        header_kolom_ketiga = df.columns[2]
        df["data"] = df.iloc[:, 2] 
        point_layer = pydeck.Layer(
                     "ScatterplotLayer",
                     data=df,
                     id="map-ai",
                     get_position=["Long", "Lat"],
                     get_color="[255, 75, 75]",
                     pickable=True,
                     auto_highlight=True,
                     get_radius="size",
                     )
        view_state = pydeck.ViewState( latitude=lat_first, longitude=lon_last, controller=True, zoom=6 )
        chart = pydeck.Deck(
                    point_layer,
                    initial_view_state=view_state,
                    tooltip={"text":"Kab/Kota: {kab}\n"+header_kolom_ketiga+": {data}"}, )
        st.pydeck_chart(chart, on_select="rerun", selection_mode="multi-object")
        st.write("***Tabel:***")
        st.write(df)

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
    st.write("***Mungkin Anda mau bertanya terkait:***")
    prompt=''
    satu, dua, tiga,empat = st.columns(4)

    if satu.button("Profil singkat Menteri Pekerjaan Umum",icon=":material/face:", use_container_width=True, type="secondary"):
        prompt= "buatkan Profil singkat Menteri Pekerjaan Umum"
    if dua.button("Peta Penanganan Perumahan Kawasan SPM Provinsi Aceh", icon=":material/public:", use_container_width=True, type="secondary"):
        prompt="Saya ingin menggabungkan tabel 7. 37 Penanganan Perumahan Kawasan SPM Provinsi Aceh untuk kolom tahun 2025 dan Tabel 21.1: Koordinat Kabupaten dan Kota menjadi satu peta dengan aturan berikut:Gabungkan berdasarkan kolom kunci Kabupaten/Kota dan nama_wilayah. Kolom gabungan harus berisi Long, Lat dan Penanganan RTLH 2025. Tolong tampilkan peta dari tabel gabungan tersebut."
    if tiga.button("Bar Chart Luasan Banjir di Madura dan Kepulauan.", icon=":material/bar_chart:", use_container_width=True, type="secondary"):
        prompt="buat bar chart Luasan Banjir di Madura dan Kepulauan dengan nama kab / kota sebagai kolom dan Luas Banjir (Ha) sebagai baris."
    if empat.button("Line Chart Proyeksi Kebutuhan Air Minum di Kawasan Simeulue", icon=":material/stacked_line_chart:",use_container_width=True, type="secondary"):
        prompt="buat line chart Proyeksi Kebutuhan Air Minum di Kawasan Simeulue dsk 2022-2034, dengan nama tahun  sebagai kolom dan populasi sebagai baris"
        
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
                time.sleep(5)
                placeholder = st.empty()
	        	# Muat potongan teks (chunk) dari file
                #text_chunks = np.load(text_chunks_path, allow_pickle=True).tolist()
                #distances, indices = search_faiss_index(prompt, index_dir, model="text-embedding-ada-002")
          		# Ambil teks asli yang relevan
                relevant_texts = search_multiple_indices(prompt, index_dir)
                #relevant_texts = retrieve_text_from_indices(indices, text_chunks)
           	 	# Gabungkan hasil teks relevan untuk diberi ke GPT-4
                context = "\n".join(relevant_texts)
            #query grafik
            if "bar chart" in prompt or "line chart" in prompt or "peta" in prompt:
                try:
                    st.header("*****Visualisasi :*****")
                    prompt_grafik = (
                        """
                            If the query requires creating a bar chart, reply as follows:
                            {"bar": {"columns": ["A", "B", "C", ...], "data": [2501, 2489, 10, ...]}}
                            If the query requires creating a line chart, reply as follows:
                            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
                            If the query requires creating a map, reply as follows:
                            {"map": {"Lat": [ .......... ], "Long": [..........], "data": [.............],"kab":[.........]}}         
                            All strings in "columns" list and data list, should be in double quotes,
                            "Lat" and "Long" columns are latitude and longitude data should following the reference Table 21.1: Koordinat Kabupaten dan Kota,
                            "Lat" and "Long" list should be in float,
                            data list should be in integer or float,
                            underscore symbol must remove from data list,
                            Lets think step by step
                            Below is the query.
                            Query:
                            """
                    + prompt
                    ) 
                    response = ask_gpt4(prompt_grafik, context, model="gpt-4o-mini")
                    decoded_response = decode_response(response)
                    write_response(decoded_response)
                except Exception as e:
                    st.error(f"Gagal memuat visualisasi. Coba Lagi!")
            else:
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

