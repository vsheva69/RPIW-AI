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
import sqlite3
from login_script import authenticate_user
from dotenv import load_dotenv
load_dotenv()

# Masukkan API key OpenAI Anda
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Database Setup ===
DB_FILE = "DB/chat_sessions.db"

def init_db():
    """Inisialisasi tabel dalam SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_name TEXT UNIQUE,
            user_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prompt TEXT,
            response TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    """)
    conn.commit()
    conn.close()

def add_user_to_db(user_id, username):
    """Tambahkan pengguna ke database lokal jika belum ada."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO users (user_id, username) VALUES (?, ?)", (user_id, username))
    conn.commit()
    conn.close()

def add_session(user_id, session_name):
    """Tambahkan sesi baru ke database untuk pengguna tertentu."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sessions (user_id, session_name)
        VALUES (?, ?)
    """, (user_id, session_name))
    conn.commit()
    conn.close()

def get_sessions(user_id):
    """Ambil semua sesi untuk pengguna tertentu."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE user_id = ?", (user_id,))
    sessions = cursor.fetchall()
    conn.close()
    return sessions

def get_session_id(user_id, session_name):
    """Ambil ID sesi berdasarkan nama dan user_id."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM sessions WHERE user_id = ? AND session_name = ?
    """, (user_id, session_name))
    session = cursor.fetchone()
    conn.close()
    return session[0] if session else None

def add_message(session_id, user_query, gpt_response):
    """Tambahkan pesan ke sesi di database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (session_id, prompt, response)
        VALUES (?, ?, ?)
    """, (session_id, user_query, gpt_response))
    conn.commit()
    conn.close()

def get_messages(session_id):
    """Ambil semua pesan dari sesi tertentu."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, prompt, response
        FROM messages
        WHERE session_id = ?
    """, (session_id,))
    messages = cursor.fetchall()
    conn.close()
    return messages


# === Fungsi Pencarian Gabungan ===
@st.cache_resource
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

    # Inisialisasi Database
    init_db()

    # Autentikasi Pengguna
    if "user" not in st.session_state:
        st.session_state.user = None

    if st.session_state.user is None:
        st.subheader("Login dengan Akun Kepegawaian")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state.user = user
                add_user_to_db(user["user_id"], user["username"])
                st.success(f"Selamat datang, {user['username']}!")
                time.sleep(2.00)
                st.rerun(scope="app")
            else:
                st.error("Login gagal! Periksa username atau password.")
        return

    # Sidebar untuk manajemen sesi
    #st.sidebar.header(f"Manajemen Sesi: {st.session_state.user['username']}")
    sessions = get_sessions(st.session_state.user["user_id"])
    session_names = [s[1] for s in sessions]

   #kolom
    one,two,three,four= st.sidebar.columns(4)
    popover = one.popover(" ", icon=":material/chat:", use_container_width=True)
    session_name = popover.text_input("Nama Percakapan", value=f"Percakapan-{len(session_names) + 1}")
    if popover.button("Buat Percakapan Baru"):
        add_session(st.session_state.user["user_id"], session_name)
        popover.success(f"Sesi '{session_name}' dibuat!")
        time.sleep(2.00)
        st.rerun(scope="app")
    profil = two.popover(" ", icon=":material/face:", use_container_width=True)
    profil.write("ini profil")
    if three.button(" ", icon=":material/refresh:", use_container_width=True):
        st.rerun(scope="app")
   # session_name = st.sidebar.text_input("Nama Sesi", value=f"Sesi-{len(session_names) + 1}")
    if four.button(" ", icon=":material/logout:", use_container_width=True):
        st.session_state.user = None
        time.sleep(2.00)
        st.rerun(scope="app")


    selected_session = st.sidebar.selectbox("**Riwayat Percakapan**",session_names )
    session_id = get_session_id(st.session_state.user["user_id"], selected_session)

    # Main content area for displaying chat messages
    st.title("Chat dengan Dokumen RPIW menggunakan AI🤖")
    st.subheader(f"Chat Sesi: {selected_session}")
    st.write("Model: GPT-4o-Mini")

    # Chat input
    # Placeholder for chat messages
    with st.expander('Disclaimer',icon="🚨"):
        st.write('''
            Informasi yang disajikan dalam chat ini merupakan hasil analisis AI berdasarkan dokumen asli. Meskipun AI berusaha memberikan informasi yang akurat, kemungkinan masih terdapat ketidaktepatan. Oleh karena itu, hasil analisis ini tidak dapat dijadikan acuan yang sepenuhnya dapat dipertanggungjawabkan kebenarannya.
        ''')
    # Riwayat sesi
    if session_id:
        messages = get_messages(session_id)
        if messages:
            for msg in messages:
                    with st.chat_message("user"):
                        st.write(f"{msg[1]}")
                    with st.chat_message("assistant"):
                        if "bar chart" in msg[1] or "line chart" in msg[1] or "peta" in msg[1]:
                            try:
                                load= msg[2]
                                decoded_response = decode_response(load)
                                write_response(decoded_response)
                            except Exception as e:
                                st.error(f"Gagal memuat visualisasi.")
                        else:
                            st.write(f"{msg[2]}")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Selamat Datang. ***Apa yang ingin anda  tanyakan?***"}]
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

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt) 
    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            # Ambil teks asli yang relevan
            relevant_texts = search_multiple_indices(prompt, index_dir)
            # Gabungkan hasil teks relevan untuk diberi ke GPT-4
            context = "\n".join(relevant_texts)
            #query grafik
            with st.spinner("Sedang berpikir...."):
                if session_id is None:
                    session_namess=f"Percakapan-{len(session_names) + 1}"
                    add_session(st.session_state.user["user_id"], session_namess)
                    session_id = get_session_id(st.session_state.user["user_id"], session_namess)
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
                            response_grafik = ask_gpt4(prompt_grafik, context, model="gpt-4o-mini")
                            decoded_response = decode_response(response_grafik)
                            write_response(decoded_response)
                            # Simpan ke database
                            if session_id:
                               add_message(session_id, prompt, response_grafik)
                            response ="***{Visualisasi}***"
                        except Exception as e:
                            st.error(f"Gagal memuat visualisasi. Coba Lagi!")
                else:
                    response = ask_gpt4(prompt, context, model="gpt-4o-mini")
                    def stream_data():
                        for word in response.split(" "):
                            yield word + " "
                            time.sleep(0.06)
                    # Simpan ke database
                    if session_id:
                       add_message(session_id, prompt, response)
                    st.write_stream(stream_data) 

        if response is not None:
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()

