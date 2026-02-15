import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="News Classifier", page_icon="📰")

# 1. Load and Filter Data
@st.cache_data
def load_bbc_data():
    # Adding encoding='latin1' usually solves the UnicodeDecodeError for this dataset
    df = pd.read_csv('bbc_data.csv', encoding='latin1')
    df = df[df['category'].isin(['sport', 'politics'])]
    return df

df = load_bbc_data()

# 2. Sidebar Setup
st.sidebar.header("Model Configuration")
technique = st.sidebar.selectbox("Select ML Technique", 
                                 ["Naive Bayes", "Logistic Regression", "SVM"])

# 3. Model Logic
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['category'], test_size=0.2, random_state=42
)

# Define Model Dictionary
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear', probability=True)
}

# Build Pipeline with TF-IDF
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', models[technique])
])

# Train
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 4. Main UI
st.title("🏆 Sport vs 🏛️ Politics Classifier")
st.markdown(f"Currently using: **{technique}** | Accuracy: **{accuracy:.2%}**")

user_input = st.text_area("Paste a news snippet here:", placeholder="The prime minister announced...")

if st.button("Classify Text"):
    if user_input:
        prediction = pipeline.predict([user_input])[0]
        prob = pipeline.predict_proba([user_input])
        
        col1, col2 = st.columns(2)
        col1.metric("Predicted Category", prediction.upper())
        col2.metric("Confidence", f"{np.max(prob):.2%}")
    else:
        st.warning("Please enter some text.")

# 5. Show Metrics Comparison (For your report)
if st.checkbox("Show Detailed Analysis"):
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred))
    st.write("Dataset Distribution:")
    st.bar_chart(df['category'].value_counts())