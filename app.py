import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Politics vs Sport Classifier", page_icon="📰")

# 1. Safe Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv('bbc_data.csv')
    # Filter for Sport and Politics based on your CSV's 'type' column
    df = df[df['type'].isin(['sport', 'politics'])]
    return df

try:
    df = load_data()
    X = df['news']
    y = df['type']
except Exception as e:
    st.error(f"Error loading CSV: {e}. Ensure 'bbc_data.csv' has 'news' and 'type' columns.")
    st.stop()

# 2. Sidebar for Model Selection
st.sidebar.title("Model Settings")
model_type = st.sidebar.selectbox("Select ML Technique", ("Naive Bayes", "Logistic Regression", "SVM"))

# 3. Training the Pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True, kernel='linear')
}

# Feature Representation: TF-IDF with Unigrams and Bigrams
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('clf', models[model_type])
])

pipeline.fit(X_train, y_train)

# 4. App UI
st.title("🏆 Sport vs 🏛️ Politics Classifier")
st.write(f"Currently testing: **{model_type}**")

user_text = st.text_area("Enter text to classify:", "The minister signed the new bill into law today.")

if st.button("Classify"):
    prediction = pipeline.predict([user_text])[0]
    st.subheader(f"Prediction: {prediction.upper()}")
    
    # Show internal stats
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    st.info(f"Model Test Accuracy: {acc:.2%}")

st.divider()
st.write("### Dataset Analysis")
st.bar_chart(df['type'].value_counts())