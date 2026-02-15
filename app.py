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

st.set_page_config(page_title="Politics vs Sport Classifier", page_icon="🏛️")

# 1. Loading Data with Correct Encoding
@st.cache_data
def load_data():
    # 'latin1' handles the special characters like the £ symbol
    df = pd.read_csv('bbc_data.csv', encoding='latin1')
    # Use the columns found in your CSV: 'news' and 'type'
    df = df[df['type'].isin(['sport', 'politics'])]
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# 2. Setup
st.title("🏆 Sport vs 🏛️ Politics Classifier")
st.sidebar.header("Model Parameters")
selected_model = st.sidebar.selectbox("Select ML Algorithm", ["Naive Bayes", "Logistic Regression", "SVM"])

# 3. Model Training Pipeline
X_train, X_test, y_train, y_test = train_test_split(df['news'], df['type'], test_size=0.2, random_state=42)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(kernel='linear', probability=True)
}

# Feature Representation: TF-IDF with Unigrams and Bigrams
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', models[selected_model])
])

pipeline.fit(X_train, y_train)

# 4. User Interface
user_input = st.text_area("Enter a news snippet:", "The election results were announced by the parliament.")

if st.button("Classify"):
    prediction = pipeline.predict([user_text] if 'user_text' in locals() else [user_input])[0]
    st.subheader(f"Result: {prediction.upper()}")
    
    # Show Probability
    probs = pipeline.predict_proba([user_input])[0]
    classes = pipeline.classes_
    prob_df = pd.DataFrame({'Category': classes, 'Probability': probs})
    st.bar_chart(prob_df.set_index('Category'))

# 5. Model Evaluation
if st.checkbox("Show Model Metrics"):
    y_pred = pipeline.predict(X_test)
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred))