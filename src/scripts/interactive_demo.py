import joblib
import os
import sys

def main():
    print("==================================================")
    print(" PathoIntern Interactive Demo - Phase 6")
    print(" Ali Cihan Ozdemir (Student ID: 9091405)")
    print("==================================================")
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'classifiers')
    try:
        vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        svd = joblib.load(os.path.join(model_dir, 'svd_model.pkl'))
        lr_svd = joblib.load(os.path.join(model_dir, 'lr_svd_model.pkl'))
    except FileNotFoundError:
        print("Models not found. Please ensure the notebook has been executed.")
        return

    print("Type your query in English or French. Type 'exit' or 'quitter' to quit.")
    while True:
        try:
            text = input("\nUser> ")
            if text.strip().lower() in ['exit', 'quitter']:
                break
            if not text.strip():
                continue
                
            text_tfidf = vectorizer.transform([text])
            text_svd = svd.transform(text_tfidf)
            
            # Use predict_proba for confidence thresholding
            proba = lr_svd.predict_proba(text_svd)[0]
            max_prob = max(proba)
            pred = lr_svd.predict(text_svd)[0]
            
            if max_prob < 0.60:
                print("-> Intent: Ambiguous Query (Confidence below 60%)")
                print("-> Action: Requesting clarification from user.")
            elif pred == 1:
                print(f"-> Intent: [1] Medical/Pathology (Confidence: {max_prob:.2f})")
                print("-> Action: Verified Medical Intent. Initializing PathoIntern Core LLM...")
            else:
                print(f"-> Intent: [0] General/System (Confidence: {max_prob:.2f})")
                print("-> Action: Handling with Lightweight Standard Response (LLM Bypassed).")
                
        except KeyboardInterrupt:
            print("\nExiting.")
            break
            
if __name__ == "__main__":
    main()
