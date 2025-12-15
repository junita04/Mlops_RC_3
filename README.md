# 🚀 MLOps Machine Failure Prediction System

## 📌 Deskripsi Proyek
Proyek ini merupakan implementasi **Machine Learning Operations (MLOps)** untuk membangun sistem prediksi kerusakan mesin berbasis data sensor. Sistem ini dirancang untuk memprediksi **kapan mesin akan mengalami kegagalan (failure cycle)** serta **persentase probabilitas kerusakan**, kemudian menampilkannya melalui sebuah website interaktif.

Proyek dikembangkan sebagai bagian dari **Tugas Besar Mata Kuliah Machine Learning Operations**, dengan fokus pada penerapan pipeline end-to-end mulai dari pengolahan data, training model, experiment tracking, hingga deployment.

---

## 🎯 Tujuan
- Membangun model machine learning untuk mendeteksi potensi kerusakan mesin berdasarkan data sensor
- Mengimplementasikan pipeline MLOps yang terstruktur dan reproducible
- Melakukan tracking eksperimen dan pemilihan model terbaik menggunakan **MLflow**
- Menyediakan interface website untuk inference model menggunakan **Streamlit**

---

## 📂 Dataset
Dataset yang digunakan berasal dari Kaggle:

🔗 **NASA Turbofan Jet Engine Data Set (C-MAPSS)**  
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

Dataset ini merupakan dataset simulasi degradasi mesin jet turbofan (C-MAPSS) yang dikembangkan oleh NASA dan banyak digunakan untuk kasus **predictive maintenance** dan **Remaining Useful Life (RUL) prediction**.

Karakteristik dataset:
- Data sensor mesin jet turbofan (run-to-failure)
- Data berbentuk tabular (berasal dari file .txt yang sudah di ekstrak)
- Memuat beberapa sensor dan kondisi operasi mesin
- Digunakan untuk memprediksi kegagalan mesin berdasarkan cycle

> ⚠️ Pada tahap ini, sistem masih menggunakan **dataset awal** dan belum menerapkan mekanisme penambahan dataset baru maupun retraining incremental.

---

## 🧠 Alur Sistem (Workflow)
1. **Data Ingestion**  
   Dataset sensor dimuat dari sumber Kaggle dan disiapkan sebagai data utama.

2. **Preprocessing Data**  
   Pembersihan data, pemilihan fitur, dan persiapan data untuk proses training.

3. **Train-Test Split**  
   Dataset dibagi menjadi:
   - 80% data training
   - 20% data testing

4. **Model Training**  
   Model machine learning dilatih untuk memprediksi:
   - Cycle terjadinya kerusakan mesin
   - Probabilitas atau persentase risiko kerusakan

5. **Model Evaluation**  
   Model dievaluasi menggunakan data testing untuk mengukur performa.

6. **Experiment Tracking (MLflow)**  
   Setiap eksperimen training dicatat menggunakan MLflow, meliputi:
   - Parameter model
   - Metric evaluasi
   - Artifact model

7. **Deployment (Streamlit)**  
   Model terbaik dideploy ke aplikasi web berbasis Streamlit untuk melakukan prediksi secara interaktif.

---

## 🛠️ Tools & Teknologi
- **Python** - Bahasa pemrograman utama yang digunakan untuk pengembangan pipeline machine learning dan MLOps.
- **Scikit-learn / Machine Learning Library** - Digunakan untuk membangun, melatih, dan mengevaluasi model machine learning berbasis data sensor.
- **MLflow** – experiment tracking & model management
- **Streamlit** – deployment dan antarmuka web
- **Pandas & NumPy** – pengolahan data
- **joblib** - Digunakan untuk menyimpan dan memuat model machine learning atau objek Python secara efisien.
- **pyyaml** - Digunakan untuk membaca dan mengelola file konfigurasi (YAML), seperti pengaturan parameter model dan pipeline.
- **matplotlib** - Digunakan untuk visualisasi data dan hasil evaluasi model dalam bentuk grafik statis.
- **seaborn** - Digunakan untuk visualisasi data statistik yang lebih informatif dan eksploratif berbasis Matplotlib.
- **plotly** - Digunakan untuk visualisasi data interaktif yang ditampilkan pada aplikasi web.

---

## 🌐 Fitur Website
- Input data sensor mesin
- Prediksi:
  - Estimasi cycle mesin akan berhenti atau rusak
  - Persentase probabilitas kerusakan
- Tampilan hasil prediksi secara real-time melalui website

---

## 📊 Experiment Tracking dengan MLflow
MLflow digunakan untuk mencatat seluruh proses training model, membandingkan performa antar eksperimen, serta menyimpan model terbaik yang digunakan pada tahap deployment. Dengan MLflow, seluruh eksperimen dapat direproduksi dan dianalisis kembali.

---

## 📌 Catatan Pengembangan
- Pipeline dirancang agar dapat dikembangkan menjadi **continuous training** pada tahap selanjutnya

---

## 👥 Tim 
- Akmal Faiz Abdillah (122450114)
- Elok Fiola (1224500
- Rut Junita Sari Siburian (122450103)
- Syalaisha Andina Putriansyah (122450111)
---

## 📄 Lisensi
Proyek ini dibuat untuk keperluan akademik dan pembelajaran.
