import streamlit as st
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Multi-Format Classifier", page_icon="📂")

# --- Helper Functions for File Extraction ---
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('bbc_data.csv', encoding='latin1')
    df = df[df['type'].isin(['sport', 'politics'])]
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading BBC dataset: {e}")
    st.stop()

# --- Model Training ---
X_train, X_test, y_train, y_test = train_test_split(df['news'], df['type'], test_size=0.2, random_state=42)

st.sidebar.title("Model Settings")
model_choice = st.sidebar.selectbox("Choose Technique", ("Naive Bayes", "Logistic Regression", "SVM"))

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(kernel='linear', probability=True)
}

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', models[model_choice])
])
pipeline.fit(X_train, y_train)

# --- Main UI ---
st.title("🏆 Sport vs 🏛️ Politics Document Classifier")
st.write("Upload a **TXT, PDF, or DOCX** file to classify it.")

uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    with st.spinner('Extracting text...'):
        try:
            if file_type == 'txt':
                content = uploaded_file.read().decode("latin1")
            elif file_type == 'pdf':
                content = extract_text_from_pdf(uploaded_file)
            elif file_type == 'docx':
                content = extract_text_from_docx(uploaded_file)
            
            st.success("File processed successfully!")
            
            with st.expander("View Extracted Text"):
                st.write(content[:1000] + "...")

            if st.button("Classify Document"):
                prediction = pipeline.predict([content])[0]
                probs = pipeline.predict_proba([content])[0]
                
                st.subheader(f"Prediction: {prediction.upper()}")
                
                # Confidence Chart
                prob_df = pd.DataFrame({
                    'Category': pipeline.classes_,
                    'Confidence': probs
                }).set_index('Category')
                st.bar_chart(prob_df)
                
        except Exception as e:
            st.error(f"Error processing file: {e}")