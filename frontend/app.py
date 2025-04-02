<<<<<<< HEAD
=======



























import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # M1/M2 chip

import multiprocessing as mp
mp.set_start_method("fork", force=True)


>>>>>>> a4d9f9d (Updated modules and folder structure)
import asyncio
import nest_asyncio
nest_asyncio.apply()

<<<<<<< HEAD
=======
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
>>>>>>> a4d9f9d (Updated modules and folder structure)

import os
from dotenv import load_dotenv
import streamlit as st
from backend.ranking import hybrid_search, rank_retrieved_documents
<<<<<<< HEAD
from backend.AImodel import query_deepseek, extract_relevant_info
=======
from backend.AImodel import load_model, query_ai, extract_relevant_info
from backend.retrieval import db
>>>>>>> a4d9f9d (Updated modules and folder structure)
from backend.consistency import consistency_check
import speech_recognition as sr
from PIL import Image
import pytesseract
<<<<<<< HEAD
import re


=======
import logging
from backend.fetcher import fetch_all_sources

# Load model once
model, tokenizer = load_model()

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.info("Starting retrieval process...")
>>>>>>> a4d9f9d (Updated modules and folder structure)

# Load API keys
load_dotenv()
hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")

<<<<<<< HEAD
# Streamlit UI with improved layout
st.set_page_config(page_title="AI Software Engineering Assistant", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: url("https://www.cio.com/wp-content/uploads/2024/09/3509174-0-20524600-1726135378-shutterstock_2041424264.jpg?resize=2048%2C1365&quality=50&strip=all") 
                        no-repeat center center fixed;
            background-size: cover;
        }

        .block-container {
            background: rgba(255, 255, 255, 0.85); 
            padding: 20px;
            border-radius: 10px;
        }

        .stTextArea textarea {
            color: #333333;
            font-size: 16px;
        }

        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #45a049 !important;
            box-shadow: 2px 2px 10px rgba(0, 255, 0, 0.5);
        }

        .retrieved-card {
            background-color: #F5F5F5;
            color: #333333;
            padding: 12px;
            border-radius: 10px;
            margin: 5px 0;
            box-shadow: 2px 2px 10px rgba(255,255,255,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; font-size: 60px; font-weight: bold; color: #4169E1; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5);'>AI Software Engineering Assistant</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='font-size: 20px; font-weight: bold; text-align: center; color: #007BA7;'>"
    "Retrieve real-time updates on coding best practices, frameworks, and DevOps."
    "</p>",
    unsafe_allow_html=True
)

# User input options
input_mode = st.radio("Choose input mode:", ["Text", "Speech", "Upload Image"])

# For text input
if input_mode == "Text":
    query = st.text_area("Enter your query:", "What is the best practice for CI/CD?")

# For speech input
=======
# Streamlit UI Setup
st.set_page_config(page_title="AI Software Engineering Assistant", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: url("https://www.cio.com/wp-content/uploads/2024/09/3509174-0-20524600-1726135378-shutterstock_2041424264.jpg?resize=2048%2C1365&quality=50&strip=all")
                    no-repeat center center fixed;
        background-size: cover;
    }
    .block-container {
        background: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
    }
    .stTextArea textarea {
        color: #333333;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049 !important;
        box-shadow: 2px 2px 10px rgba(0, 255, 0, 0.5);
    }
    .retrieved-card {
        background-color: #F5F5F5;
        color: #333333;
        padding: 12px;
        border-radius: 10px;
        margin: 5px 0;
        box-shadow: 2px 2px 10px rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 60px; font-weight: bold; color: #4169E1; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5);'>AI Software Engineering Assistant</h1>", unsafe_allow_html=True)

st.markdown("""
<p style='font-size: 20px; font-weight: bold; text-align: center; color: #007BA7;'>
Retrieve real-time updates on coding best practices, frameworks, and DevOps.
</p>
""", unsafe_allow_html=True)

# Input Mode Selection
input_mode = st.radio("Choose input mode:", ["Text", "Speech", "Upload Image"])
query = "Generate a set of code snippets for implementing a web-based user interface for a software system that allows users to manage their personal finances."

if input_mode == "Text":
    query = st.text_area("Enter your query:")
>>>>>>> a4d9f9d (Updated modules and folder structure)
elif input_mode == "Speech":
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Speak your query now...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        st.success(f"🎤 Recognized Speech: {query}")
    except sr.UnknownValueError:
        st.error("Could not understand the speech.")
<<<<<<< HEAD
        query = ""

# For image upload (OCR-based)
=======
>>>>>>> a4d9f9d (Updated modules and folder structure)
elif input_mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image containing text:", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        query = pytesseract.image_to_string(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"📝 Extracted Text: {query}")

<<<<<<< HEAD
# Generate response when the button is clicked
if st.button("Generate Response"):
    with st.spinner("🔄 Fetching relevant knowledge..."):
        retrieved_docs = hybrid_search(query, k=3)

        if not retrieved_docs:
            st.warning("⚠️ No relevant documents found. AI may generate responses without supporting knowledge.")

        ranked_docs = rank_retrieved_documents(retrieved_docs) if retrieved_docs else []

    # Display Retrieved Knowledge
    st.subheader("📖 Retrieved Knowledge")

    if ranked_docs:
        for doc in ranked_docs:
            with st.container():
                extracted_summary = extract_relevant_info([doc], query=query)
            st.markdown(
                f"<div class='retrieved-card'>🔹 {extracted_summary}...<br><small>📌 Source: {doc.get('source', 'Unknown')}</small></div>",
                unsafe_allow_html=True,
            )
    else:
        st.write("No relevant documents retrieved.")

    # Generate AI Response
    with st.spinner("🤖 Generating AI response..."):
        response = query_deepseek(query, ranked_docs, hugging_face_token)

    # Display AI Response
    st.subheader("💡 AI Response")
    st.markdown(f"<div class='retrieved-card'>{response}</div>", unsafe_allow_html=True)

    # Consistency Check
    similarity_score = consistency_check(query, response)
    if similarity_score is False:
        st.warning("⚠️ Inconsistency detected, manual review needed!")
    else:
        st.success(f"✅ Response is consistent with retrieved knowledge (Similarity: {similarity_score:.2f})")

    from backend.retrieval import db  # Import ChromaDB instance

    # DEBUG: Check if database has data
    if db._collection.count() == 0:
        st.warning("⚠️ The AI knowledge database is empty! Fetching new sources...")
        from backend.arxiv_fetcher import fetch_latest_articles  # Import fetch function
        articles = fetch_latest_articles()
        
        # Index these articles in the vector database
        for article in articles:
            db.add_texts([article["title"] + " " + article["source"]], metadatas=[article])
        
        st.success(f"✅ Successfully added {len(articles)} new knowledge items!")































############################ CODE WITHOUT SPEECH AND IMAGE UPLOAD FUNCTION ######
# import os
# from dotenv import load_dotenv
# import streamlit as st
# from ranking import hybrid_search, rank_retrieved_documents
# from AImodel import query_deepseek
# from consistency import consistency_check

# # Load API keys
# load_dotenv()
# hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")

# # Streamlit UI with improved layout
# st.set_page_config(page_title="AI Software Engineering Assistant", layout="wide")

# st.markdown(
#     """
#     <style>
#         /* Set Background Image for the Entire App */
#         .stApp {
#             background: url("https://www.cio.com/wp-content/uploads/2024/09/3509174-0-20524600-1726135378-shutterstock_2041424264.jpg?resize=2048%2C1365&quality=50&strip=all") 
#                         no-repeat center center fixed;
#             background-size: cover;
#         }

#         /* Optional: Add transparency to content */
#         .block-container {
#             background: rgba(255, 255, 255, 0.85);  /* White with 85% opacity */
#             padding: 20px;
#             border-radius: 10px;
#         }
        
#         /* Change actual text color inside the input field */
#         .stTextArea textarea {
#             color: #333333;  /* Change text color */
#             font-size: 16px;
#         }


#         /* 3️⃣ Styling the Button */
#         .stButton>button {
#             background-color: #4CAF50 !important; /* Green button */
#             color: white !important;
#             padding: 12px 24px;
#             font-size: 16px;
#             font-weight: bold;
#             border: none;
#             border-radius: 8px;
#             box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
#             transition: 0.3s;
#         }

#         .stButton>button:hover {
#             background-color: #45a049 !important;
#             box-shadow: 2px 2px 10px rgba(0, 255, 0, 0.5);
#         }

#         /* Retrieved Cards */
#         .retrieved-card {
#             background-color: #F5F5F5;
            
#             color: #333333;
#             padding: 12px;
#             border-radius: 10px;
#             margin: 5px 0;
#             box-shadow: 2px 2px 10px rgba(255,255,255,0.1);
#         }
#     </style>

#     """,
#     unsafe_allow_html=True
# )


# st.markdown("<h1 style='text-align: center; font-size: 60px; font-weight: bold; color: #4169E1; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5);'>AI Software Engineering Assistant</h1>", unsafe_allow_html=True)
# #st.markdown("<p class='info'>Retrieve real-time updates on coding best practices, frameworks, and DevOps.</p>", unsafe_allow_html=True)
# st.markdown(
#     "<p style='font-size: 20px; font-weight: bold; text-align: center; color: #007BA7;'>"
#     "Retrieve real-time updates on coding best practices, frameworks, and DevOps."
#     "</p>",
#     unsafe_allow_html=True
# )

# # Sidebar for query input
# query = st.text_area("Enter your query:", "What is the best practice for CI/CD?")

# if st.button("Generate Response"):
#     with st.spinner("🔄 Fetching relevant knowledge..."):
#         retrieved_docs = hybrid_search(query, k=3)

#         if not retrieved_docs:
#             st.warning("⚠️ No relevant documents found. AI may generate responses without supporting knowledge.")

#         ranked_docs = rank_retrieved_documents(retrieved_docs) if retrieved_docs else []

#     # Display Retrieved Knowledge
#     st.subheader("📖 Retrieved Knowledge")

#     if ranked_docs:
#         for doc in ranked_docs:
#             with st.container():
#                 st.markdown(
#                     f"<div class='retrieved-card'>🔹 {doc.page_content[:200]}...<br><small>📌 Source: {doc.metadata.get('source', 'Unknown')}</small></div>",
#                     unsafe_allow_html=True,
#                 )
#     else:
#         st.write("No relevant documents retrieved.")

#     # Step 2: Generate AI Response
#     with st.spinner("🤖 Generating AI response..."):
#         response = query_deepseek(query, ranked_docs, hugging_face_token)

#     # Step 3: Display AI Response
#     st.subheader("💡 AI Response")
#     st.markdown(f"<div class='retrieved-card'>{response}</div>", unsafe_allow_html=True)

#     # Step 4: Consistency Check
#     similarity_score = consistency_check(query, response)
#     if similarity_score is False:
#         st.warning("⚠️ Inconsistency detected, manual review needed!")
#     else:
#         st.success(f"✅ Response is consistent with retrieved knowledge (Similarity: {similarity_score:.2f})")


=======
if st.button("Generate Response") and query:
    with st.spinner("🔄 Fetching relevant knowledge..."):
        retrieved_docs = hybrid_search(query, k=5)
        ranked_docs = rank_retrieved_documents(retrieved_docs) if retrieved_docs else []

    st.subheader("📖 Retrieved Knowledge")
    # if ranked_docs:
    #     for doc in ranked_docs:
    #         summary = extract_relevant_info([doc], query=query)
    #         source = doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else doc.get("source", "Unknown")
    #         st.markdown(
    #             f"<div class='retrieved-card'>🔹 {summary}<br><small>📌 Source: {source}</small></div>",
    #             unsafe_allow_html=True
    #         )

    # Deduplicate based on content
    seen_texts = set()
    unique_docs = []

    for doc in ranked_docs:
        text = doc.page_content if hasattr(doc, "page_content") else doc.get("text", "")
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_docs.append(doc)

    # Now show only unique ones
    if unique_docs:
        for doc in unique_docs:
            summary = extract_relevant_info([doc], query=query)
            source = doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else doc.get("source", "Unknown")
            st.markdown(
                f"<div class='retrieved-card'>🔹 {summary}<br><small>📌 Source: {source}</small></div>",
                unsafe_allow_html=True
            )





    else:
        st.warning("No relevant documents retrieved.")

    with st.spinner("🤖 Generating AI response..."):
        
        response = query_ai(query, ranked_docs, model, tokenizer)

    is_consistent = consistency_check(query, response)

    st.subheader("💡 AI Response")
    if is_consistent:
        st.success("✅ Consistent response")
    else:
        st.warning("⚠️ Inconsistency detected. Manual review suggested.")

    st.markdown(f"<div class='retrieved-card'>{response}</div>", unsafe_allow_html=True)

    articles = fetch_all_sources()
    added = 0

    for article in articles:
        title = article.get("title") or article.get("text") or "Untitled"
        source = article.get("source", "Unknown")
        try:
            db.add_texts([f"{title} {source}"], metadatas=[article])
            added += 1
        except Exception as e:
            st.warning(f"⚠️ Skipped an article due to error: {e}")

    st.success(f"✅ Successfully added {added} new knowledge items!")



import atexit

@atexit.register
def cleanup():
    import multiprocessing as mp
    try:
        mp.resource_tracker.unregister('/dev/shm', 'semaphore')
    except Exception:
        pass
>>>>>>> a4d9f9d (Updated modules and folder structure)
