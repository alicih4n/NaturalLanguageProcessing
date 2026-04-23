import os
import joblib
import pytest

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'classifiers')

@pytest.fixture(scope="module")
def models():
    # Load models
    vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    svd = joblib.load(os.path.join(MODEL_DIR, 'svd_model.pkl'))
    lr_svd = joblib.load(os.path.join(MODEL_DIR, 'lr_svd_model.pkl'))
    return vectorizer, svd, lr_svd

def predict(text, models):
    vectorizer, svd, lr_svd = models
    if not text.strip():
        return 0
    text_tfidf = vectorizer.transform([text])
    text_svd = svd.transform(text_tfidf)
    return lr_svd.predict(text_svd)[0]

@pytest.mark.parametrize("query", [
    "Hello",
    "How are you?",
    "Who built you?",
    "Where is the lab?"
])
def test_general_intent(models, query):
    """Test Case A (General): MUST return Label 0."""
    pred = predict(query, models)
    assert pred == 0, f"Expected 0 for query: {query}, but got {pred}"

@pytest.mark.parametrize("query", [
    "carcinoma histology",
    "blood smear anomaly",
    "leukocyte count",
    "Analyze this lab slide"
])
def test_medical_intent(models, query):
    """Test Case B (Medical): MUST return Label 1."""
    pred = predict(query, models)
    assert pred == 1, f"Expected 1 for query: {query}, but got {pred}"

@pytest.mark.parametrize("query", [
    "Bonjour",
    "C'est quoi ce projet?"
])
def test_french_support(models, query):
    """Test Case C (French Support): MUST return Label 0."""
    pred = predict(query, models)
    assert pred == 0, f"Expected 0 for query: {query}, but got {pred}"

def test_empty_string(models):
    """Empty String Test: Ensure the pipeline handles empty/whitespace inputs gracefully."""
    empty_queries = ["", "   ", "\n\t"]
    for q in empty_queries:
        pred = predict(q, models)
        assert pred == 0
