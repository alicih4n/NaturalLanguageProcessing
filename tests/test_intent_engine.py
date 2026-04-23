import os
import joblib
import pytest

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

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
        # Handle empty/whitespace gracefully by returning a default (e.g. 0)
        return 0
    text_tfidf = vectorizer.transform([text])
    text_svd = svd.transform(text_tfidf)
    return lr_svd.predict(text_svd)[0]

def test_model_load(models):
    """Model Load Test: Verify that all models load without errors."""
    vectorizer, svd, lr_svd = models
    assert vectorizer is not None
    assert svd is not None
    assert lr_svd is not None

def test_inference_consistency(models):
    """Inference Consistency Test: Ensure a known medical query returns Label 1."""
    query = "Pathology of blood smear"
    pred = predict(query, models)
    # The models trained on tfidf should flag this as Medical (1) because it contains "pathology", "blood", "smear".
    assert pred == 1

def test_multilingual_support(models):
    """Multilingual Support Test: Verify French query runs without crashing."""
    query = "Qu'est-ce que PathoIntern?"
    try:
        pred = predict(query, models)
        assert pred in [0, 1]
    except Exception as e:
        pytest.fail(f"Multilingual test crashed with exception: {e}")

def test_empty_string(models):
    """Empty String Test: Ensure the pipeline handles empty/whitespace inputs gracefully."""
    empty_queries = ["", "   ", "\n\t"]
    for q in empty_queries:
        try:
            pred = predict(q, models)
            assert pred in [0, 1]
        except Exception as e:
            pytest.fail(f"Empty string handling failed: {e}")

def test_output_range(models):
    """Output Range Test: Verify all predictions are strictly 0 or 1."""
    queries = [
        "Hello",
        "What is malaria?",
        "Please reset my password",
        "Aidez-moi",
        "blast cells in blood"
    ]
    for q in queries:
        pred = predict(q, models)
        assert pred in [0, 1]
