
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from huggingface_hub import login

hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
login(token=hf_token)
#from backend.consistency import embedding_model
# Trust scores for different sources
source_trust_scores = {
    "arxiv.org": 0.9,
    "github.com": 0.8,
    "stackoverflow.com": 0.7,
    "reddit.com": 0.3}  # Less reliable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#from langchain_community.vectorstores import Chroma
# new correct imports explicitly:
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from backend.fetcher import fetch_all_sources
from backend.consistency import embedding_model
# embedding model clearly defined
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
# explicitly fetch sources
documents = fetch_all_sources()

# texts and metadata explicitly matched
texts = [doc["text"] for doc in documents]
metadatas = [{"source": doc["source"], "link": doc["link"]} for doc in documents]

# explicitly create new Chroma DB (not updating old one!)
db = Chroma.from_texts(
    texts=texts,
    embedding=embedding_model,
    metadatas=metadatas,
    persist_directory="./chroma_db"
)

#print("✅ Vector database explicitly rebuilt successfully!")


# Trust scores for different sources
source_trust_scores = {
    "arxiv.org": 0.3,
    "github.com": 0.9,
    "stackoverflow.com": 0.7,
    "reddit.com": 0.1,  # Less reliable
    "dzone.com": 0.9, 
    "medium.com": 0.8,
    "scholarly": 0.5
}

# def get_source_trust(source):
#     """Assigns trust score based on the source URL."""
#     for key in source_trust_scores:
#         if key in source:
#             return source_trust_scores[key]
#     return 0.5  # Default trust score
def get_source_trust(source):
    trust = {"ArXiv": 0.6, "GitHub": 0.8, "Stack Overflow": 0.8, "RFCs": 0.9}

    trust = {"ArXiv": 0.6, "GitHub": 0.9, "Stack Overflow": 0.9, "RFCs": 0.85}

    return trust.get(source, 0.5)

def rank_retrieved_documents(documents):
    """Ranks documents dynamically based on trust, recency, and relevance."""
    for doc in documents:
        # Example recency scoring: if document text contains "2024" or "2025", boost it
        recency_score = 1.0 if ("2024" in doc["text"] or "2025" in doc["text"]) else 0.5
        trust_score = get_source_trust(doc["source"])
        # Combine scores (you can adjust weights as needed)
        doc["final_score"] = (trust_score * 0.5 + recency_score * 0.5) * 0.5 + doc.get("relevance_score", 0.5) * 0.5
    return sorted(documents, key=lambda x: x["final_score"], reverse=True)




# Initialize FAISS-based vector search

#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# BM25-based search with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')  # Using English stopwords by default, or you can set stop_words=None to disable it

def compute_bm25_scores(query, documents):
    """Computes BM25-like relevance scores using TF-IDF similarity."""
    # Filter out documents that are empty or contain only stop words
    documents = [doc for doc in documents if doc["text"].strip()]
    
    if not documents:  # If all documents are empty, return an empty list
        return documents
    
    corpus = [doc["text"] for doc in documents]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    for i, doc in enumerate(documents):
        doc["relevance_score"] = scores[i]
    
    return documents

def hybrid_search(query, k=5):
    """Combines BM25 (TF-IDF) with vector search for better retrieval."""
    vector_results = db.similarity_search(query, k=k)
    
    # Convert vector results to document format
    documents = [
        {"text": doc.page_content, "source": doc.metadata.get("source", "unknown"), "link": doc.metadata.get("link", "#")} 
        for doc in vector_results
    ]

    # Compute BM25 scores
    documents = compute_bm25_scores(query, documents)

    # Rank documents
    ranked_docs = rank_retrieved_documents(documents)
    
    return ranked_docs[:k]

# Example usage:
retrieved_docs = hybrid_search("latest AI frameworks")
for doc in retrieved_docs:

    print(f"🔹 {doc['text'][:100]}... (Source: {doc['source']}, Link: {doc['link']})")






    print(f"🔹 {doc['text'][:300]}... (Source: {doc['source']}, Link: {doc['link']})")

