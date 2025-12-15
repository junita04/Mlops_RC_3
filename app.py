import re
import joblib
import streamlit as st

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load("tfidf.pkl")
    model = joblib.load("model_lr.pkl")
    return tfidf, model

tfidf, model = load_artifacts()

def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.set_page_config(page_title="Fake Job Detector", page_icon="🛡️")
st.title("🛡️ Fake / Scam Job Posting Detector")
st.write("Paste teks lowongan → cek apakah **legit** atau **scam**.")

text_in = st.text_area(
    "Teks lowongan (title/description/requirements/benefits)",
    height=220,
    placeholder="Paste di sini..."
)

threshold = st.slider("Threshold scam", 0.10, 0.90, 0.50, 0.05)

if st.button("🔍 Check", type="primary"):
    cleaned = clean_text(text_in)
    if len(cleaned) < 20:
        st.warning("Teks terlalu pendek. Paste deskripsi yang lebih lengkap ya.")
    else:
        X_vec = tfidf.transform([cleaned])
        proba_scam = float(model.predict_proba(X_vec)[0, 1])
        pred = 1 if proba_scam >= threshold else 0

        if pred == 1:
            st.error("⚠️ POTENSI SCAM / FAKE")
        else:
            st.success("✅ TERLIHAT LEGIT")

        st.write(f"Probabilitas scam: **{proba_scam:.3f}**")
        st.caption("Gunakan sebagai filter awal, bukan keputusan final.")
