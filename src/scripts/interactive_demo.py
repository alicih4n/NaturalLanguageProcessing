import joblib
import os
import sys
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the LlamaManager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.chatbot_engine import LlamaManager

def detect_language(query: str) -> str:
    french_indicators = {"le", "la", "les", "des", "un", "une", "est", "qui", "quoi", "bonjour", "projet", "membre", "c'est", "comment", "salut", "merci"}
    words = set(query.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "").split())
    if words.intersection(french_indicators):
        return "fr"
    return "en"

def retrieve_context(query: str, language: str):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if language == "fr":
        kb_path = os.path.join(base_dir, 'data', 'raw', 'knowledge_base_fr.json')
    else:
        kb_path = os.path.join(base_dir, 'data', 'raw', 'knowledge_base_en.json')
        
    if not os.path.exists(kb_path):
        return "", 0.0
        
    with open(kb_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    questions = []
    answers = []
    for item in data.get('qa_pairs', []):
        questions.append(item['question'])
        answers.append(item['answer'])
        
    if not questions:
        return "", 0.0
        
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(questions)
    query_vec = vectorizer.transform([query])
    
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = sims.argmax()
    
    return answers[best_idx], sims[best_idx]

def main():
    print("==================================================")
    print(" PathoIntern Interactive Hybrid RAG Demo")
    print(" Ali Cihan Ozdemir (Student ID: 9091405)")
    print("==================================================")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_dir = os.path.join(base_dir, 'models', 'classifiers')
    
    try:
        vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        svd = joblib.load(os.path.join(model_dir, 'svd_model.pkl'))
        lr_svd = joblib.load(os.path.join(model_dir, 'lr_svd_model.pkl'))
    except FileNotFoundError:
        print("Models not found. Please ensure the notebook has been executed.")
        return

    # Eager Loading the LLM
    print("\n[System] Pre-loading 200MB GGUF Model into RAM for zero-latency inference...")
    llama = LlamaManager()
    llama.load_model()
    print("[System] Model loaded successfully.")
    
    print("\nType your query in English or French. Type 'exit' or 'quitter' to quit.")
    while True:
        try:
            text = input("\nUser> ")
            if text.strip().lower() in ['exit', 'quitter']:
                break
            if not text.strip():
                continue
                
            detected_language = detect_language(text)
                
            text_tfidf = vectorizer.transform([text])
            text_svd = svd.transform(text_tfidf)
            
            # Use predict_proba for confidence thresholding
            proba = lr_svd.predict_proba(text_svd)[0]
            max_prob = max(proba)
            pred = lr_svd.predict(text_svd)[0]
            
            if max_prob < 0.60:
                print("-> Intent: Ambiguous Query (Confidence below 60%)")
                print("-> Action: Requesting clarification from user.")
                if detected_language == "fr":
                    print("\n🤖 System: Je ne suis pas sûr de comprendre. Pourriez-vous clarifier ?")
                else:
                    print("\n🤖 System: I'm not entirely sure what you mean. Could you please clarify?")
            elif pred == 1:
                print(f"-> Intent: [1] Medical/Project (Confidence: {max_prob:.2f})")
                print(f"-> Detected Language: {detected_language.upper()}")
                
                context, sim_score = retrieve_context(text, detected_language)
                
                if sim_score > 0.85:
                    print(f"-> Action: Hyper-Precise Match (Score: {sim_score:.2f}). Bypassing LLM.")
                    print(f"\n🤖 PathoIntern (Direct JSON Match): {context}")
                elif sim_score >= 0.4:
                    print(f"-> Action: Medium Confidence Match (Score: {sim_score:.2f}). Routing to GGUF LLM...")
                    try:
                        response = llama.generate_rag_response(query=text, context=context, language=detected_language)
                        print(f"\n🤖 PathoIntern (GGUF): {response}")
                    except Exception as e:
                        print(f"\n🤖 Error running LLM: {e}")
                else:
                    print(f"-> Action: Weak Context (Score: {sim_score:.2f}). Rejecting Query.")
                    if detected_language == "fr":
                        print("\n🤖 System: Je ne possède pas d'informations spécifiques à ce sujet dans ma base de connaissances.")
                    else:
                        print("\n🤖 System: I do not have specific information about this in my knowledge base.")
            else:
                print(f"-> Intent: [0] General/System (Confidence: {max_prob:.2f})")
                print(f"-> Detected Language: {detected_language.upper()}")
                print("-> Action: Handling with Lightweight Standard Response (LLM Bypassed).")
                if detected_language == "fr":
                    print("\n🤖 System: Bonjour! Je suis l'assistant système PathoIntern. Je peux vous aider à naviguer dans le système ou répondre à des questions de base.")
                else:
                    print("\n🤖 System: Hello! I am the PathoIntern system assistant. I can help you navigate the system or answer basic questions.")
                
        except KeyboardInterrupt:
            print("\nExiting.")
            break
            
if __name__ == "__main__":
    main()
