import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embedding model
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
# Initialize ChromaDB with embedding function
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

def consistency_check(query, new_response, threshold=0.6):

    """Checks consistency of new response by comparing it with past responses."""
    
    # Retrieve past responses from the database
    past_responses = db.similarity_search(query, k=3)
    
    # If no past responses exist, assume consistency
    if not past_responses:
        print("ℹ️ No past responses found. Assuming consistency.")
        return True
    
    past_texts = [doc.page_content for doc in past_responses]
    
    # Convert responses to vector representation
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(past_texts + [new_response])
    
    # Compute cosine similarity between new response and past responses
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    
    # Check similarity threshold
    max_sim = np.max(cosine_sim)
    if max_sim < threshold:
        print(f"⚠️ Possible knowledge drift detected! Max similarity: {max_sim:.2f}")
        return False
    
    print(f"✅ Response is consistent. Max similarity: {max_sim:.2f}")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load embedding model

#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

def generate_structured_answer(query, retrieved_docs):
    """Create a clean, structured answer from retrieved documents."""
    knowledge_lines = []
    for doc in retrieved_docs:
        try:
            content = doc.page_content  # LangChain Document
        except AttributeError:
            content = doc.get("page_content", "")
        knowledge_lines.append(f"- {content.strip()}")
    
    knowledge = "\n".join(knowledge_lines)

    return (
        f"You are an expert software engineering assistant.\n\n"
        f"### User Query ###\n{query}\n\n"
        f"### Extracted Knowledge ###\n{knowledge}\n\n"
        f"### Instructions ###\n"
        f"For each module, include:\n"
        f"1. Name\n2. Functionality\n3. Challenges\n4. Interactions\n"
        f"Then recommend a development order.\n\n"
        f"### AI Answer ###\n"
    )



def consistency_check(query, new_response, threshold=0.2):
    """Checks consistency of new response by comparing it with past responses."""
    past_responses = db.similarity_search(query, k=5)
    if not past_responses:
        print("ℹ️ No past responses found. Assuming consistency.")
        return True

    past_texts = [doc.page_content for doc in past_responses]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(past_texts + [new_response])
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

    max_sim = np.max(cosine_sim)
    if max_sim < threshold:
        print(f"⚠️ Possible knowledge drift detected! Max similarity: {max_sim:.3f}")
        return False

    print(f"✅ Response is consistent. Max similarity: {max_sim:.3f}")
    return True
