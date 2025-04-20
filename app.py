import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {'not', 'but'}
stemmer = PorterStemmer()

# Load model & vectorizer
model = joblib.load("news_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Map categories
category_map = {
    "World": "🌍 World",
    "Sports": "🏅 Sports",
    "Business": "📈 Business",
    "Science/Tech": "🧪 Science/Tech"
}
category_desc = {
    "World": "International news & global affairs.",
    "Sports": "Matches, tournaments, and sports events.",
    "Business": "Markets, economy, and companies.",
    "Science/Tech": "Innovations, AI, and discoveries."
}

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return " ".join(stemmer.stem(word) for word in text.split() if word not in stop_words)

# Predict function
def predict_news(texts):
    cleaned = [clean_text(t) for t in texts]
    tfidf = vectorizer.transform(cleaned)
    probs = model.predict_proba(tfidf)
    preds = model.predict(tfidf)
    return preds, np.max(probs, axis=1)

# UI
st.set_page_config(page_title="🗞️ News Classifier", layout="wide", initial_sidebar_state="expanded")

st.title("🗞️ News Category Classifier")
st.markdown("Classify news headlines into categories using a trained ML model.")

st.sidebar.title("⚙️ Input")
sample_texts = st.sidebar.text_area("🔤 Enter News Headlines (one per line)", height=150)

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV file with 'Title' column", type=["csv"])

# Main content area
if uploaded_file or sample_texts:
    with st.spinner("Classifying..."):
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            texts = df["Title"].dropna().tolist()
        else:
            texts = sample_texts.strip().split("\n")

        predictions, confidences = predict_news(texts)

        result_df = pd.DataFrame({
            "Headline": texts,
            "Category": predictions,
            "Confidence": confidences
        })
        result_df["Icon"] = result_df["Category"].map(category_map)
        result_df["Confidence (%)"] = (result_df["Confidence"] * 100).round(2)
        result_df["Short"] = result_df["Headline"].apply(lambda x: x[:70] + "..." if len(x) > 70 else x)

        st.subheader("📊 Results")
        for idx, row in result_df.iterrows():
            col1, col2 = st.columns([6, 3])
            with col1:
                st.markdown(f"**{row['Icon']}** {row['Short']}")
                st.caption(category_desc.get(row["Category"], ""))
            with col2:
                color = "green" if row["Confidence"] > 0.75 else "orange" if row["Confidence"] > 0.5 else "red"
                st.progress(row["Confidence"], text=f"{row['Confidence (%)']}% confident")

        # CSV download
        csv = result_df[["Headline", "Category", "Confidence (%)"]].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", csv, "news_predictions.csv", "text/csv")

else:
    st.info("👈 Enter headlines or upload a file to get started!")

st.markdown("---")
st.caption("Made with ❤️ by Aakash · Powered by Naive Bayes & Streamlit")
