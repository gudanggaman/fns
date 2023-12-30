# Import library Streamlit untuk membuat aplikasi web sederhana
import streamlit as st
# Library untuk menyimpan dan memuat model serta data
import pickle 
# Library untuk manipulasi data numerik
import numpy as np 
# Library untuk ekstraksi fitur teks
from sklearn.feature_extraction.text import TfidfVectorizer
# Library untuk model klasifikasi Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# Fungsi ini membaca model yang telah disimpan untuk ekstraksi fitur teks
def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkt", "rb"))
    return tfidf

# Fungsi ini membaca model yang telah disimpan untuk klasifikasi Naive Bayes
def load_model():
    nb_model = pickle.load(open("toxicity_model.pkt", "rb"))
    return nb_model

# Fungsi ini melakukan prediksi apakah suatu teks bersifat toksik atau tidak
def toxicity_prediction(text):
    # Menggunakan model ekstraksi fitur teks yang telah dibaca sebelumnya
    tfidf = load_tfidf()
    # Mengubah teks input menjadi vektor fitur menggunakan model TF-IDF
    text_tfidf = tfidf.transform([text]).toarray()
    # Menggunakan model klasifikasi Naive Bayes yang telah dibaca sebelumnya
    nb_model = load_model()
    # Melakukan prediksi apakah teks bersifat toksik atau tidak
    prediction = nb_model.predict(text_tfidf)
    # Menentukan kategori hasil prediksi (Toksik atau Non-Toksik)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name

# Bagian ini menggunakan Streamlit untuk membuat antarmuka web
st.header("Toxicity Detection App")

# Bagian ini menampilkan subjudul dan input teks pada antarmuka web
st.subheader("Input your text")
text_input = st.text_input("Enter your text")

# Bagian ini menambahkan tombol untuk menganalisis teks
if text_input is not None:
    if st.button("Analyse"):
        # Bagian ini melakukan prediksi dan menampilkan hasil pada antarmuka web
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        st.info("The result is "+ result + ".")
