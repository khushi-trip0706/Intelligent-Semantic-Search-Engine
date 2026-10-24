import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config

def load_knowledge_base(file_path = config.DATA_FILE_PATH):
    """Dynamically reads lines from the file designed in config."""
    if not os.path.exists(file_path):
        default_data = [
            "Project MedTrack is managed via Solidity smart contracts.",
            "Project AerialRestore uses OpenCV to remove atmospheric haze.",
            "Project Watchdog is a Deep Learning Anomaly Detection System running Tensorflow."
        ]
        
        with open(file_path,"w") as f:
            f.write("\n".join(default_data))
        return default_data
    
    with open(file_path, "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines() if line.strip()]
    return documents

def run_interactive_search():
    documents = load_knowledge_base()
    
    print("--- Matrix-Based Search Engine Pipeline Live ---")
    print(f"[ENGINE LOG] Successfully indexed {len(documents)} documents from {config.DATA_FILE_PATH}.\n")
    
    vectorizer = TfidfVectorizer(lowercase=config.LOWERCASE_PROCESSING)
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    while True:
        print('-' * 60)
        user_query = input(f"Type your search query (or type '{config.EXIT_COMMAND}' to quit):").strip()
        
        if user_query.lower() == config.EXIT_COMMAND:
            print("Shutting down the search engine. Goodbye!")
            break
        
        if not user_query:
            print("[ERROR] Query cannont be empty. Please try again.")
            continue
        
        query_vector = vectorizer.transform([user_query])
        similarity_score = cosine_similarity(query_vector,tfidf_matrix).flatten()
        
        best_match_index = np.argmax(similarity_score)
        highest_score = similarity_score[best_match_index]
        
        print("\n--- [AI ENGINE MATCHING RESULT] ---")
        if highest_score > 0.0:
            print(f"Best Matching Document Found(Match Confidence: {highest_score:.4f}):")
            print(f">> {documents[best_match_index]}\n")
        else:
            print("[WARNING LOG] No matching context could be found for that query.\n")

if __name__ == "__main__":
    run_interactive_search()