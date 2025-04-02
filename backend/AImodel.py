
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
from dotenv import load_dotenv
import os
from huggingface_hub import login
import wandb

# Load API keys from .env file
load_dotenv()
hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")
wnb_token = os.getenv("WANDB_API_KEY")

if not hugging_face_token or not wnb_token:
    raise ValueError("API keys not found! Check your .env file.")

# Login to Hugging Face
login(hugging_face_token)

# Login to Weights & Biases
wandb.login(key=wnb_token)


## mistralai/Mistral-7B-Instruct

def load_model(hugging_face_token, model_name="openai-community/gpt2-xl"):  
    # Load tokenizer with authentication token
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hugging_face_token)

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  

    # Explicitly add a special padding token
import multiprocessing as mp
mp.set_start_method("fork", force=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be.*")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSeq2SeqLM
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
import wandb
import re
from peft import PeftModel

# ------------------------
# Environment Setup
# ------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")
wandb_token = os.getenv("WANDB_API_KEY")

if not hugging_face_token or not wandb_token:
    raise ValueError("❌ Missing HUGGINGFACE_TOKEN or WANDB_API_KEY in .env")

login(hugging_face_token)
wandb.login(key=wandb_token)

# ------------------------
# Load Model
# ------------------------
def load_model(
    base_model="openai-community/gpt2-xl",
    lora_weights_path="./output",
    use_lora=False
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model with authentication token
    model = AutoModelForCausalLM.from_pretrained(base_model, token=hugging_face_token).to(device)

    # Resize model embeddings to match the updated tokenizer
    model.resize_token_embeddings(len(tokenizer))

    print(f"Model and tokenizer loaded successfully on {device}!")
    return model, tokenizer


# def load_model(hugging_face_token):
#     model_name = "openai-community/gpt2"
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         token=hugging_face_token,
#         trust_remote_code=True  # Allow custom code to be executed
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     return model, tokenizer

import re

def extract_relevant_info(documents, query=""):
    """Extract key sentences related to the user query from retrieved documents."""
    key_sentences = []
    for doc in documents:
        text = doc.get("text", "")
        if not text.strip():
            continue
        # Split text into sentences
        sentences = re.split(r'(?<=\.)\s+', text)
        # Filter sentences that contain any keywords from the query
        keywords = query.lower().split()
        filtered = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
        # If no sentence contains the keywords, use the first sentence as fallback
        if not filtered and sentences:
            filtered = [sentences[0]]
        key_sentences.extend(filtered[:2])  # take up to 2 sentences per document
    if not key_sentences:
        return "No content available."
    return " ".join(key_sentences[:5])


def query_deepseek(query, documents, model, tokenizer):
    """Improved response generation using summarized context."""
    #model, tokenizer = load_model(hugging_face_token)
    #model, tokenizer= load_model()
    
    if not documents:
        return "I couldn't find any relevant information on this topic. Please refine your query."

    # Extract key insights instead of full documents
    context_summary = extract_relevant_info(documents)

    input_text = (
    "You are an AI assistant specialized in software engineering. Your answer MUST be based solely on the extracted knowledge provided below. "
    "Do NOT generate generic statements. Follow this structure exactly:\n\n"
    "### User Query ###\n" + query + "\n\n"
    "### Extracted Knowledge ###\n" + context_summary + "\n\n"
    "### Answer Format ###\n"
    "- **Key Trends:** List 3 important trends mentioned in the knowledge.\n"
    "- **Why It Matters:** Explain why these trends are significant.\n"
    "- **Industry Adoption:** Provide real-world examples.\n\n"
    "### AI Answer ###")





    print("DEBUG: AI is receiving this input:\n" + input_text + "\n")
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True) 

    # Move inputs to correct device
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Generate response
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],  
        max_length=500,  
        pad_token_id=tokenizer.pad_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    if use_lora:
        print(f"🔧 Loading LoRA adapter from {lora_weights_path}")
        model = PeftModel.from_pretrained(model, lora_weights_path)

    model.resize_token_embeddings(len(tokenizer))
    print(f"✅ CodeLlama model and tokenizer loaded on {device} (LoRA: {use_lora})")
    return model, tokenizer

# ------------------------
# Contextual Extraction
# ------------------------
def extract_relevant_info(documents, query=""):
    keywords = set(query.lower().split())
    relevant_sentences = []

    for doc in documents:
        text = doc.page_content if hasattr(doc, "page_content") else doc.get("text", "")
        sentences = re.split(r'(?<=\\.)\\s+', text)
        matches = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
        if matches:
            relevant_sentences.extend(matches[:5])
        elif sentences:
            relevant_sentences.append(sentences[0])

    if not relevant_sentences:
        return (
            "Effective software development practices include agile methodologies, continuous integration, "
            "automated testing, code reviews, and frequent feedback loops to ensure quality and adaptability."
        )

    return ' '.join(relevant_sentences[:15])

# ------------------------
# Main Query Function
# ------------------------
def query_ai(user_query, retrieved_docs, model, tokenizer):
    from transformers import logging
    logging.set_verbosity_error()

    # Deduplicate docs
    seen = set()
    unique_docs = []
    for doc in retrieved_docs:
        text = doc.page_content if hasattr(doc, "page_content") else doc.get("text", "")
        if text and text not in seen:
            seen.add(text)
            unique_docs.append(doc)

    # Build knowledge string
    knowledge_lines = [
    f"- {(doc.page_content if hasattr(doc, 'page_content') else doc.get('text', '')).strip()[:300]} "
    f"(Source: {doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else doc.get('source', 'Unknown')})"
    for doc in unique_docs[:5]]

    knowledge = "\\n".join(knowledge_lines)

    if not knowledge.strip():
        knowledge = "No high-confidence sources were retrieved. Proceed using best practices."

    # Build CodeLlama-style prompt
    prompt = (
        f"<s> You are a software engineering assistant. Given the following user query and extracted knowledge, "
        f"provide a structured breakdown of components/modules for the project.\n\n"
        f"User Query:\n{user_query}\n\n"
        f"Extracted Knowledge:\n{knowledge}\n\n"
        #f"For each module, include:\n"
        #f"1. Name\n2. Functionality\n3. Implementation Challenges\n4. Interactions\n"
        #f"Finally, suggest a logical development order."
    )

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=786,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True
    )

    # decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # return decoded.strip()
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    decoded = re.sub(r"(?i)user query.*", "", decoded).strip()
    cleaned = re.sub(r"[~*_>`#]", "", decoded)  # clean markdown-sensitive characters
    return cleaned.strip()
