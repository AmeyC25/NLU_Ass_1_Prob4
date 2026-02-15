import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Data Loading (Using subsets of 20 Newsgroups for Sports vs Politics)
@st.cache_resource
def load_data():
    categories = ['rec.sport.hockey', 'rec.sport.baseball', 'talk.politics.mideast', 'talk.politics.misc']
    data_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    data_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    # Map to binary: 0 for Sports, 1 for Politics
    y_train = [0 if 'sport' in data_train.target_names[i] else 1 for i in data_train.target]
    y_test = [0 if 'sport' in data_test.target_names[i] else 1 for i in data_test.target]
    
    return data_train.data, y_train, data_test.data, y_test

X_train, y_train, X_test, y_test = load_data()

# 2. Model Training
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True)
}

st.title("🏆 Sports vs 🏛️ Politics Classifier")
st.write("Comparing Naive Bayes, Logistic Regression, and SVM using TF-IDF.")

selected_model_name = st.selectbox("Choose a Model to Test", list(models.keys()))

# Build Pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
    ('clf', models[selected_model_name]),
])

text_clf.fit(X_train, y_train)

# 3. UI for Testing
user_input = st.text_area("Enter text to classify:", "The striker scored a goal in the final minute.")

if st.button("Classify"):
    pred = text_clf.predict([user_input])[0]
    label = "Sports" if pred == 0 else "Politics"
    st.subheader(f"Result: {label}")
    
    # Show Accuracy
    y_pred = text_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.info(f"Model Test Accuracy: {acc:.2%}")