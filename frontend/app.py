import os
import sys
import logging
import asyncio
import atexit
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import speech_recognition as sr
import nest_asyncio
import multiprocessing as mp

# ==== Streamlit Config ====
st.set_page_config(page_title="AI Software Engineering Assistant", layout="wide")

# ==== System Paths ====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ==== Async Setup ====
nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

mp.set_start_method("fork", force=True)

# ==== Imports ====
from huggingface_hub import login
from backend.ranking import hybrid_search, rank_retrieved_documents
from backend.AImodel import load_model, query_ai, query_deepseek, extract_relevant_info
from backend.retrieval import db
from backend.consistency import consistency_check
from backend.fetcher import fetch_all_sources

# ==== Logging ====
logging.basicConfig(level=logging.INFO)
logging.info("Starting retrieval process...")

# ==== API Keys & Model Loading ====
load_dotenv()
hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")
login(hugging_face_token)
model, tokenizer = load_model()
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # M1/M2 chip

# ==== Streamlit Styling ====
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

# ==== App Title ====
st.markdown("<h1 style='text-align: center; font-size: 60px; font-weight: bold; color: #4169E1; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5);'>AI Software Engineering Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 20px; font-weight: bold; text-align: center; color: #007BA7;'>Retrieve real-time updates on coding best practices, frameworks, and DevOps.</p>", unsafe_allow_html=True)

# ==== User Input ====
input_mode = st.radio("Choose input mode:", ["Text", "Speech", "Upload Image"], key="input_mode")
query = ""

if input_mode == "Text":
    query = st.text_area("Enter your query:", "What is the best practice for CI/CD?", key="text_input")

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
        query = ""

elif input_mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image containing text:", type=["png", "jpg", "jpeg"], key="image_upload")
    if uploaded_image:
        image = Image.open(uploaded_image)
        query = pytesseract.image_to_string(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.success(f"📝 Extracted Text: {query}")

# ==== Process Query ====
if st.button("Generate Response", key="generate_btn") and query:
    with st.spinner("🔄 Fetching relevant knowledge..."):
        retrieved_docs = hybrid_search(query, k=3)
        ranked_docs = rank_retrieved_documents(retrieved_docs) if retrieved_docs else []

    # ==== Display Retrieved Knowledge ====
    st.subheader("📖 Retrieved Knowledge")
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



# ==== Cleanup ====

import atexit

@atexit.register
def cleanup():
    import multiprocessing as mp
    try:
        mp.resource_tracker.unregister('/dev/shm', 'semaphore')
    except Exception:
        pass
