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

    print("Type your query in English or French. Type 'exit' to quit.")
    while True:
        try:
            text = input("\nUser> ")
            if text.strip().lower() == 'exit':
                break
            if not text.strip():
                continue
                
            text_tfidf = vectorizer.transform([text])
            text_svd = svd.transform(text_tfidf)
            pred = lr_svd.predict(text_svd)[0]
            
            if pred == 1:
                print("-> Intent: [1] Medical/Pathology")
                print("-> Action: Routing to Local GGUF Model (LFM2.5-350M-Q4_K_M.gguf)...")
            else:
                print("-> Intent: [0] General/System")
                print("-> Action: Handling with Lightweight Standard Response (LLM Bypassed).")
                
        except KeyboardInterrupt:
            print("\nExiting.")
            break
            
if __name__ == "__main__":
    main()
