# 🏠 RoomEase MVP

RoomEase is an AI-powered roommate and apartment recommendation app built with **Streamlit**, **Gemini 2 Flash**, and **local LLaMA (via LM Studio)**. It matches students with compatible roommates and ideal apartments in Madrid based on preferences, lifestyle, and budget.

---

## ✨ Features

* 💬 **Conversational interface** – Streamlit chatbot guides the user through 10 personality questions.
* 🧠 **AI-based roommate matching** – Uses local embeddings and Gemini Flash 2 for personality compatibility.
* 🏢 **Apartment finder** – Combines semantic search, reranking, and structured filters via Gemini.
* 🔍 **RAG architecture** – Uses ChromaDB + SentenceTransformer embeddings for student and apartment indexing.
* ⚡ **Local + cloud hybrid** – Runs Gemini API in combination with a local LLaMA model served through LM Studio.

---

## 🧱 Tech Stack

* **Frontend:** Streamlit (Python)
* **LLMs:** LLaMA 3B Instruct (local via LM Studio), Gemini Flash 2 (Google Generative AI)
* **Vector DB:** ChromaDB (Persistent client)
* **Embeddings:** SentenceTransformer (`all-MiniLM-L6-v2`)
* **Re-ranking:** CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
* **PDF Parsing:** PyPDF (student metadata extraction)

---

## 🚀 Getting Started

### 1️⃣ Prerequisites

* Python ≥ 3.10
* [LM Studio](https://lmstudio.ai/) installed and running
* Download **LLaMA 3B Instruct** model inside LM Studio
* A valid **Gemini API key** for Google’s Generative AI

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/ARM-02/roomease-mvp.git
cd roomease-mvp
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Setup

### 1️⃣ Start LM Studio Server

1. Open **LM Studio**
2. Download and load **LLaMA 3B Instruct** model
3. Start a **local API server** at port **1234**

   * The endpoint should look like: `http://127.0.0.1:1234/v1/chat/completions`

### 2️⃣ Set Gemini API Key

```bash
export GEMINI_API_KEY="your_gemini_flash2_api_key"
```

*(For Windows PowerShell: `$env:GEMINI_API_KEY="your_gemini_flash2_api_key"`)*

---

## ▶️ Running the App

Once LM Studio and the environment are set up:

```bash
streamlit run app.py
```

Then open the Streamlit link (usually `http://localhost:8501`) in your browser.

---

## 🗂️ Project Structure

```
roomease-mvp/
├── app.py                       # Streamlit front-end chatbot
├── rag_backend.py               # Core logic for apartment & roommate recommendations
├── extract_students_metadata.py # Extracts structured data from student PDFs via LLM
├── embed_index.py               # Embeds apartment & student data into ChromaDB
├── apartment_description_summarizer.py # Summarizes Idealista listings via local LLM
├── chroma_store/                # Persistent ChromaDB collections
├── data/                        # Datasets (PDFs, CSVs)
└── requirements.txt             # Dependencies
```

---

## 🧪 Example Workflow

1. **Index apartments:**

   ```bash
   python embed_index.py --csv data/available_apartments.csv --reset-apartments
   ```
2. **Index student profiles:**

   ```bash
   python extract_students_metadata.py --pdf data/student_profiles.pdf --reset
   ```
3. **Run the chat app:**

   ```bash
   streamlit run app.py
   ```

---

## 🧠 Architecture Overview

* **LM Studio (LLaMA)** → Summarizes long apartment descriptions locally.
* **Gemini Flash 2** → Parses user queries, filters data, and scores compatibility.
* **ChromaDB** → Stores embeddings for apartments and students.
* **SentenceTransformer + CrossEncoder** → Handles retrieval and reranking.

---

### 🧩 Quick TL;DR

1. Download LLM Studio → Load LLaMA 3B Instruct → Start server on port 1234
2. `export GEMINI_API_KEY=...`
3. `streamlit run app.py`
   ✅ That’s it — RoomEase is live!
