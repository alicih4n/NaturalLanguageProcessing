import json
import os
import random
import csv

def load_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [item['question'] for item in data['qa_pairs']]

def augment_medical_questions(questions, target_count=1000):
    augmented = []
    prefixes_en = ["Can you tell me: ", "I need to know, ", "What about: ", "Please explain ", "Query: "]
    prefixes_fr = ["Pouvez-vous me dire: ", "J'ai besoin de savoir: ", "Qu'en est-il de: ", "Veuillez expliquer ", "Question: "]
    
    # Just add the original ones first
    augmented.extend(questions)
    
    while len(augmented) < target_count:
        q = random.choice(questions)
        aug_type = random.choice(['prefix_en', 'prefix_fr', 'lower', 'no_punct', 'split'])
        if aug_type == 'prefix_en':
            q_aug = random.choice(prefixes_en) + q.lower()
        elif aug_type == 'prefix_fr':
            q_aug = random.choice(prefixes_fr) + q.lower()
        elif aug_type == 'lower':
            q_aug = q.lower()
        elif aug_type == 'no_punct':
            q_aug = q.replace('?', '').replace('.', '').replace(',', '')
        elif aug_type == 'split':
            parts = q.split()
            if len(parts) > 3:
                q_aug = " ".join(parts[:len(parts)//2]) + "... " + " ".join(parts[len(parts)//2:])
            else:
                q_aug = q
        augmented.append(q_aug)
    
    return augmented[:target_count]

def generate_general_questions(target_count=1000):
    base_general = [
        "Hello", "Hi there", "How are you?", "Good morning", "Good evening", 
        "Reset my password", "I forgot my password", "Help me", "What is your name?",
        "Are you a robot?", "Thank you", "Bye", "See you later", "Who created you?",
        "Bonjour", "Salut", "Comment ça va?", "Bonsoir", "Merci", "Au revoir",
        "Comment tu t'appelles?", "Qui es-tu?", "Tu es un robot?", "Aidez-moi",
        "Mot de passe oublié", "Réinitialiser mon mot de passe", "À plus tard",
        "Good afternoon", "Hey", "What's up?", "Can you assist me?", "I need help",
        "Yes", "No", "Maybe", "I don't know", "Please", "Thanks", "Ok", "Okay",
        "Oui", "Non", "Peut-être", "Je ne sais pas", "S'il vous plaît", "D'accord"
    ]
    
    augmented = []
    while len(augmented) < target_count:
        q = random.choice(base_general)
        aug_type = random.choice(['normal', 'lower', 'upper', 'exclaim', 'question', 'dots'])
        if aug_type == 'normal':
            q_aug = q
        elif aug_type == 'lower':
            q_aug = q.lower()
        elif aug_type == 'upper':
            q_aug = q.upper()
        elif aug_type == 'exclaim':
            q_aug = q + "!"
        elif aug_type == 'question':
            q_aug = q + "?"
        elif aug_type == 'dots':
            q_aug = q + "..."
        augmented.append(q_aug)
        
    return augmented[:target_count]

def main():
    en_questions = load_questions('knowledge_base_en.json')
    fr_questions = load_questions('knowledge_base_fr.json')
    all_medical = en_questions + fr_questions
    
    medical_data = augment_medical_questions(all_medical, 1000)
    general_data = generate_general_questions(1000)
    
    dataset = []
    for q in medical_data:
        dataset.append({'text': q, 'label': 1})
    for q in general_data:
        dataset.append({'text': q, 'label': 0})
        
    # Shuffle dataset
    random.shuffle(dataset)
    
    os.makedirs('data', exist_ok=True)
    
    with open('data/pathology_classification_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(dataset)
        
    print(f"Generated dataset with {len(dataset)} rows at data/pathology_classification_dataset.csv")

if __name__ == '__main__':
    main()
