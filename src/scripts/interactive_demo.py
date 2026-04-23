import joblib
import os
import sys
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the LlamaManager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.core.chatbot_engine import LlamaManager

def load_knowledge_base():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    kb_path_en = os.path.join(base_dir, 'data', 'raw', 'knowledge_base_en.json')
    kb_path_fr = os.path.join(base_dir, 'data', 'raw', 'knowledge_base_fr.json')
    
    questions = []
    answers = []
    
    for kb_path in [kb_path_en, kb_path_fr]:
        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data.get('qa_pairs', []):
                questions.append(item['question'])
                answers.append(item['answer'])
                
    return questions, answers

def retrieve_context(query, questions, answers):
    if not questions:
        return ""
    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(questions)
    query_vec = vectorizer.transform([query])
    
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = sims.argmax()
    
    if sims[best_idx] > 0.05: # basic threshold
        return answers[best_idx]
    return "I couldn't find specific information about this in my knowledge base, but I will try to answer."

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

    print("Loading Knowledge Base...")
    kb_questions, kb_answers = load_knowledge_base()
    
    # Initialize the LLM Manager early so it loads once if possible
    llama = LlamaManager()
    
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
                print("\n🤖 System: I'm not entirely sure what you mean. Could you please clarify?")
            elif pred == 1:
                print(f"-> Intent: [1] Medical/Pathology (Confidence: {max_prob:.2f})")
                print("-> Action: Verified Medical Intent. Retrieving context from JSON...")
                context = retrieve_context(text, kb_questions, kb_answers)
                
                print("-> Action: Initializing PathoIntern Core LLM (GGUF)...")
                try:
                    response = llama.generate_rag_response(text, context)
                    print(f"\n🤖 PathoIntern (GGUF): {response}")
                except Exception as e:
                    print(f"\n🤖 Error running LLM: {e}")
            else:
                print(f"-> Intent: [0] General/System (Confidence: {max_prob:.2f})")
                print("-> Action: Handling with Lightweight Standard Response (LLM Bypassed).")
                print("\n🤖 System: Hello! I am the PathoIntern system assistant. I can help you navigate the system or answer basic questions.")
                
        except KeyboardInterrupt:
            print("\nExiting.")
            break
            
if __name__ == "__main__":
    main()
