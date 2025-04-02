<<<<<<< HEAD
# from sentence_transformers import SentenceTransformer
# import chromadb

# # Load an embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize ChromaDB with embedding function
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_or_create_collection(name="my_collection")

# def embed_texts(texts):
#     return embedding_model.encode(texts).tolist()

# def query_deepseek(query, documents, hugging_face_token):
#     model, tokenizer = load_model(hugging_face_token)
    
#     # Convert documents to embeddings
#     doc_texts = [doc['text'] for doc in documents]
#     doc_embeddings = embed_texts(doc_texts)

#     # Store in ChromaDB (avoid duplicates)
#     for i, (doc, embedding) in enumerate(zip(doc_texts, doc_embeddings)):
#         collection.add(
#             documents=[doc], 
#             embeddings=[embedding], 
#             ids=[str(i)]  # Unique ID required
#         )

#     # Generate embedding for query
#     query_embedding = [embedding_model.encode(query).tolist()]  # Fix format

#     # Retrieve most relevant documents
#     results = collection.query(query_embeddings=query_embedding, n_results=3)

#     # Ensure results exist
#     if not results["documents"]:
#         return "No relevant documents found!"

#     relevant_docs = [doc for doc in results["documents"][0]]  # Extract document texts

#     # Generate response using model
#     context = " ".join(relevant_docs)
#     input_text = f"Query: {query}\nContext: {context}\nAnswer:"
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
#     inputs = {key: value.to(model.device) for key, value in inputs.items()}
#     outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=500)
    
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.arxiv_fetcher import fetch_all_sources

=======
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from backend.fetcher import fetch_all_sources
>>>>>>> a4d9f9d (Updated modules and folder structure)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from backend.AImodel import extract_relevant_info
<<<<<<< HEAD
# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
=======
import re 
from backend.consistency import embedding_model

# Load embedding model
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
>>>>>>> a4d9f9d (Updated modules and folder structure)
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# TF-IDF-based retrieval
vectorizer = TfidfVectorizer(stop_words='english')

# Cross-encoder model for re-ranking
<<<<<<< HEAD
reranker = CrossEncoder("BAAI/bge-reranker-large")
=======
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
>>>>>>> a4d9f9d (Updated modules and folder structure)

def compute_bm25_scores(query, documents):
    """Computes BM25-like relevance scores using TF-IDF similarity."""
    documents = [doc for doc in documents if doc["text"].strip()]

    if not documents:
        return documents
    
    corpus = [doc["text"] for doc in documents]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    for i, doc in enumerate(documents):
        doc["relevance_score"] = scores[i]

    return documents
<<<<<<< HEAD
import re

import re
=======

>>>>>>> a4d9f9d (Updated modules and folder structure)

def extract_relevant_info(documents, query=""):
    """Extracts key sentences related to the user query from retrieved documents."""
    key_sentences = []

    for doc in documents:
        text = doc.get("text", "")
        if not text.strip():
            continue
        # Split the text into sentences
        sentences = re.split(r'(?<=\.)\s+', text)
        # Filter sentences that include at least one keyword from the query
        keywords = query.lower().split()
        filtered = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
        if filtered:
            key_sentences.extend(filtered[:2])  # take up to 2 sentences per document

    if not key_sentences:
        return "No highly relevant details found. AI will generate a general response."
    return " ".join(key_sentences[:5])  # return up to 5 key sentences




def extract_key_sentences(documents, query):
    """Extract key sentences related to the query from retrieved documents."""
    import re
    key_sentences = []

    for doc in documents:
        text = doc.get("text", "")

        # Ensure valid text
        if not text.strip():
            continue

        # Split into sentences
        sentences = re.split(r'(?<=\.)\s+', text)

        # Filter sentences containing query-related keywords
        keywords = query.lower().split()
        relevant_sentences = [s for s in sentences if any(kw in s.lower() for kw in keywords)]

        if relevant_sentences:
            key_sentences.extend(relevant_sentences[:2])  # Take first 2 relevant sentences

    return " ".join(key_sentences[:5]) if key_sentences else "No highly relevant details found."


def hybrid_search(query, k=5):
    """Performs hybrid retrieval (BM25 + Vector Search + Reranking)."""
    
    # Debug: Ensure the vector database has data
    doc_count = db._collection.count()  
    print(f"🟢 Database contains {doc_count} documents.")

    if doc_count == 0:
        print("⚠️ Warning: No documents found in the retrieval database!")
    
    # Proceed with vector and keyword search
    vector_results = db.similarity_search(query, k=k*2)

    if not vector_results:
        print(f"⚠️ No vector search results for '{query}'! Trying external sources...")
        vector_results = fetch_all_sources()

        if  not vector_results:
            print("⚠️ No new articles were fetched! Check internet connection or API changes.")
        else:
            print(f"✅ Successfully fetched {len(documents)} new articles!")

<<<<<<< HEAD
    documents = [
    {"text": doc.page_content, "source": doc.metadata.get("source", "unknown"), "link": doc.metadata.get("link", "#")}
    for doc in vector_results
    ]
=======
    # documents = [
    # {"text": doc.page_content, "source": doc.metadata.get("source", "unknown"), "link": doc.metadata.get("link", "#")}
    # for doc in vector_results
    # ]
    documents = fetch_all_sources()
>>>>>>> a4d9f9d (Updated modules and folder structure)
    # Update the document text by extracting key sentences using the query:
    for doc in documents:
        doc["text"] = extract_relevant_info([doc], query=query)
    return documents[:k]





def get_source_trust(source):
    """Dynamically adjusts source trustworthiness based on user feedback, citations, and recency."""
    trust_scores = {
        "ArXiv": 0.7,  # Academic sources are usually trustworthy
        "GitHub": 0.6,  # Code examples are valuable, but unverified
        "Stack Overflow": 0.5,  # Community-driven but can be outdated
        "RFCs": 0.9,  # Official standards, high trust
    }

    return trust_scores.get(source, 0.5)  # Default trust score

def rank_retrieved_documents(documents):
    """Ranks documents dynamically based on trust, relevance, recency, and feedback."""
    for doc in documents:
        recency_score = 1.0 if "2024" in doc["text"] else 0.5  # Boost newer knowledge
        doc["trust_score"] = get_source_trust(doc["source"]) * 0.5 + recency_score * 0.5

    return sorted(
        documents,
        key=lambda doc: (doc["trust_score"] * 0.5) + (doc["relevance_score"] * 0.5),
        reverse=True
<<<<<<< HEAD
    )
=======
    )
>>>>>>> a4d9f9d (Updated modules and folder structure)
