# 🤖 AI Software Engineering Assistant
### *"How Can AI Stay Up to Date Without Constant Retraining?"*

---

## 👩‍💻 GROUP 4: AIMonvi
- **Team Members:** Seyedeh (Mona) Ebrahimi, Tanvi Sharma  
- **Project Mentor:** Ayman
- **Project Title:** *How Can AI Stay Up to Date Without Constant Retraining?*

---

## 🧭 Overview
The **AI Software Engineering Assistant** helps developers, architects, and DevOps engineers stay current by:
- 🔧 Tracking new frameworks, tools, and practices
- 📚 Automatically fetching knowledge from reliable sources
- 🧠 Generating answers using real-time data, without retraining the model

It uses:
- 🔍 Retrieval-Augmented Generation (RAG)
- 🎯 Trust-based document ranking
- ✅ Consistency checking to ensure relevance and accuracy

Sources include:
- ArXiv, GitHub, Stack Overflow, RFCs
- Hacker News, Reddit, Dev.to, Medium
- Google Scholar, GitHub Discussions

---

## 🧱 Project Structure

```plaintext
📦 Project Modules
│
├── 🧠 AImodel.py         → Loads LLM + structured generation
├── 🌐 fetcher.py         → Retrieves knowledge from 10+ real-time sources
├── 🧠 retrieval.py       → Embeds + stores docs in ChromaDB
├── 🎯 ranking.py         → Ranks docs based on trust and relevance
├── 🔍 consistency.py     → Compares with past responses to prevent drift
├── 🖼️  app.py             → Streamlit UI with text, speech, and image input
```

---

## ⚙️ Installation Instructions

### ✅ Step 1: Set Up a Virtual Environment
```bash
conda create -n capstone_env python=3.10
conda activate capstone_env
conda install pytorch torchvision torchaudio
Please add your tokens to the ".env" file before running!
```

### ✅ Step 2: Install Required Python Libraries
```bash
pip install streamlit
pip install torch transformers peft trl accelerate chromadb faiss-cpu sentence-transformers langchain rank_bm25 -U langchain-community
pip install python-dotenv wandb ipywidgets
pip install speechrecognition pydub pillow pytesseract pyaudio
pip install beautifulsoup4 scholarly
```

### 🍏 Step 3: Special Instructions for macOS (PyAudio & Tesseract)
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add Homebrew to PATH
echo 'eval "$\(/opt/homebrew/bin/brew shellenv\)"' >> ~/.zprofile
eval "$\(/opt/homebrew/bin/brew shellenv\)"

# Install audio & OCR dependencies
brew install portaudio
brew install tesseract

# Then reinstall PyAudio
pip install pyaudio
```

---

## ▶️ Step 4: Simply Run the Application 
```bash
streamlit run frontend/app.py
```
> This opens the assistant in your browser. You can enter a query, speak a question, or upload a screenshot and one gets the responses and links to the resources. 
> In addition to that, a Json file includes the link and information you searched for in the query, will be saved too! Yaaaaaaaaay:)
---

## 🔄 System Workflow

```plaintext
User Query
   ↓
Fetcher → Real-time sources (GitHub, Reddit, ArXiv...)
   ↓
Retriever → Embed + store in ChromaDB
   ↓
Ranker → Score & filter results
   ↓
AI Model → Generate answer using prompt & knowledge
   ↓
Consistency Checker → Compare to past output
   ↓
Streamlit UI → Display structured, validated response
```

---

## ✨ Key Features
- 🔄 Stays current without retraining
- 🎙️ Supports voice and image queries
- 💡 Context-aware generation
- ⚖️ Output validation with consistency scoring
- 🌍 Fully open-source + offline compatible

---

## 👨‍💻 Contributors

- [@seyedeh-mona-ebrahimi](https://github.com/seyedeh-mona-ebrahimi)
- [@tanvicat](https://github.com/tanvicat)

---

## 📄 License
MIT License

---

## 🔗 Links & Resources
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [HuggingFace](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)

> "In a world where code evolves weekly, your assistant should too."
