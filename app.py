import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Document Classifier", page_icon="📄")

@st.cache_data
def load_data():
    df = pd.read_csv('bbc_data.csv', encoding='latin1')
    df = df[df['type'].isin(['sport', 'politics'])]
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Model Selection
st.sidebar.title("Configuration")
model_type = st.sidebar.selectbox("Algorithm", ("Naive Bayes", "Logistic Regression", "SVM"))

# Training
X_train, X_test, y_train, y_test = train_test_split(df['news'], df['type'], test_size=0.2, random_state=42)
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(kernel='linear', probability=True)
}

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', models[model_type])
])
pipeline.fit(X_train, y_train)

# UI for Document Upload
st.title("📂 Sports vs Politics Document Classifier")
st.write("Upload a `.txt` file to classify its content.")

uploaded_file = st.file_uploader("Choose a text document", type="txt")

if uploaded_file is not None:
    # Read document content
    document_text = uploaded_file.read().decode("latin1")
    
    st.divider()
    st.subheader("Document Preview")
    st.text(document_text[:500] + "...") # Show first 500 chars

    if st.button("Classify Document"):
        prediction = pipeline.predict([document_text])[0]
        prob = pipeline.predict_proba([document_text])[0]
        
        st.success(f"The document is classified as: **{prediction.upper()}**")
        
        # Performance metrics for the report
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        st.info(f"Current Model Accuracy: {acc:.2%}")