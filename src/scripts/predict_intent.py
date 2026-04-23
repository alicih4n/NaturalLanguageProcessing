import joblib
import os

def predict_intent(text):
    # Load models
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'classifiers')
    try:
        vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        svd = joblib.load(os.path.join(model_dir, 'svd_model.pkl'))
        lr_svd = joblib.load(os.path.join(model_dir, 'lr_svd_model.pkl'))
    except FileNotFoundError:
        print("Models not found. Please run the notebook cells first to generate them.")
        return

    # Process text
    text_tfidf = vectorizer.transform([text])
    text_svd = svd.transform(text_tfidf)
    
    # Predict
    pred = lr_svd.predict(text_svd)[0]
    
    # Output result
    if pred == 1:
        print(f'Input: "{text}" -> Output: "Class 1 (Pathology/System) - Route to GGUF."')
    else:
        print(f'Input: "{text}" -> Output: "Class 0 (General) - Standard Response."')

if __name__ == '__main__':
    print("--- Hybrid Bridge Prediction Demo ---")
    predict_intent("What is PathoIntern?")
    predict_intent("Tell me a joke.")
    predict_intent("Reset my password please.")
    predict_intent("Bonjour! Comment ça va?")
    predict_intent("What is the anomaly baseline?")
