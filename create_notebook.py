import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # Title Cell
    title_md = """# Multilingual Intent Classification for Pathology Triage Assistant
**Ali Cihan Ozdemir (9091405)**"""

    # Step 1: Data Loading
    step1_md = """### Step 1: Data Loading
Load the generated dataset and split it into 75% Training and 25% Testing."""
    step1_code = """import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('data/pathology_classification_dataset.csv')
print("Dataset Shape:", df.shape)
print(df['label'].value_counts())

# Split into train and test
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))"""

    # Step 2: TF-IDF
    step2_md = """### Step 2: TF-IDF Vectorization
Apply TfidfVectorizer (multilingual, lowercase, remove stopwords) and visualize top 20 tokens by frequency."""
    step2_code = """import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk

# Using english and french stop words if available, otherwise just english
# For demonstration without requiring NLTK downloads in the exact environment, we pass 'english' built-in stop words.
# A more robust approach for multilingual would be to combine lists.
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Calculate sum of tfidf scores for each feature
sum_words = X_train_tfidf.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:20]

words = [w[0] for w in words_freq]
scores = [w[1] for w in words_freq]

plt.figure(figsize=(10, 6))
plt.bar(words, scores, color='skyblue')
plt.title('Top 20 Tokens by TF-IDF Frequency')
plt.xlabel('Tokens')
plt.ylabel('TF-IDF Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()"""

    # Step 3: Baseline Model
    step3_md = """### Step 3: Baseline Model - Naive Bayes
Train a MultinomialNB classifier, predict on the test set, and evaluate performance."""
    step3_code = """from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Train model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Predict
y_pred = nb_classifier.predict(X_test_tfidf)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['General (0)', 'Medical (1)'], yticklabels=['General (0)', 'Medical (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()"""

    # Markdown Analysis
    analysis_md = """### Analysis
Naive Bayes is a strong baseline for sparse TF-IDF text data because it handles high-dimensional, sparse feature spaces exceptionally well without requiring extensive computational resources. By applying the assumption of conditional independence among features (words), Multinomial Naive Bayes effectively calculates the probability of a document belonging to a specific class given its word frequencies. In this context, it may struggle slightly more with the **General (0)** class when interacting with out-of-vocabulary slang, very short ambiguous inputs, or conversational phrases that overlap with terms used in Medical queries (such as "help" or "how"). Overall, its simplicity and robustness against overfitting on sparse data make it an ideal starting point."""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(title_md),
        nbf.v4.new_markdown_cell(step1_md),
        nbf.v4.new_code_cell(step1_code),
        nbf.v4.new_markdown_cell(step2_md),
        nbf.v4.new_code_cell(step2_code),
        nbf.v4.new_markdown_cell(step3_md),
        nbf.v4.new_code_cell(step3_code),
        nbf.v4.new_markdown_cell(analysis_md)
    ]
    
    with open('PathoIntern_Intent_Engine.ipynb', 'w') as f:
        nbf.write(nb, f)
        
if __name__ == '__main__':
    create_notebook()
