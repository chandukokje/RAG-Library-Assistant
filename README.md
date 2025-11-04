# 📚 Books Q&A with FAISS + LangChain + Ollama

This project demonstrates how to build a **retrieval‑augmented generation (RAG) pipeline** for book metadata using:

- **LangChain** for document abstraction and chaining
- **FAISS** for efficient vector similarity search
- **HuggingFace** for embeddings using model 'all-MiniLM-L6-v2'
- **Ollama LLM** for natural language reasoning
- **Pandas** for data wrangling

It allows you to query a dataset of books (from a JSONL file) and get **semantic answers** enriched with metadata such as authors, decades, and ratings.

---

## 🚀 Features

- Load and normalize book metadata from JSONL
- Create **row‑level documents** (per book) and **aggregate documents** (per author, per decade, top‑rated books)
- Store embeddings in a **FAISS vector store** for fast retrieval
- Interactive **chat loop** powered by Ollama LLM
- Supports prompts like:
    1. Row‑Level Book Documents (type: Book)
        - *"Tell me about the book The System of the World.*"
        - *"What is the average rating of Cryptonomicon?*"
        - *"Who wrote Anathem and when was it published?*"

    2. Author Aggregates (type: AuthorAggregate)
        - *"How many books does Neal Stephenson have?*"
        - *"Which author has written the most books?*"
        - *"Compare the number of books by J.K. Rowling and Isaac Asimov.*"

    3. Decade Aggregates (type: DecadeAggregate)
        - *"How many books were published in the 1990s?*"
        - *"Which decade had the most books?*"
        - *"Summarize book publishing trends across decades.*"

    4. Top‑Rated Books (type: TopRated)
        - *"What are the top‑rated books overall?*"
        - *"List the highest‑rated books from the 2000s.*"
        - *"Which books have an average rating above 4.5 with more than 10,000 ratings?*"
---

## 📂 Project Structure
```
. 
├── vector.py # Builds FAISS vector store from books.jsonl 
├── main.py # Interactive Q&A loop with Ollama + retriever 
├── books.jsonl # Dataset (line‑delimited JSON with book metadata) 
└── README.md # Documentation
```

---

## 📦 Requirements

### 1. Create and activate a virtual environment
```bash
# Create virtual environment named vLibraryAssistant
python -m venv vLibraryAssistant

# Activate on Windows (PowerShell)
vLibraryAssistant\Scripts\Activate.ps1

# Install dependencies:
pip install -r .\requirements.txt

# Install Ollama model
ollama pull llama3.2
```

## 🗂 Dataset Format
The dataset (books.jsonl) should contain entries like:

``` json
json
{
  "title": "The System of the World",
  "authors": ["Neal Stephenson"],
  "publication_year": 2004,
  "id": "5707",
  "average_rating": 4.3,
  "image_url": "https://images.gr-assets.com/books/1407712273m/116257.jpg",
  "ratings_count": 16106
}
```

## ▶️ Usage
1. Build the Vector Store
Run vector.py once to load the dataset, enrich metadata, and build the FAISS index:

```bash
python vector.py
```
This creates a local folder BooksDB/ containing the FAISS index.

2. Start the Interactive Q&A
Run main.py to launch the chat loop:

```bash
python main.py
```

You’ll see:
```
📚 Book Q&A Assistant (type 'q' to quit)
-----------------------------------------

Ask your question:
```

## 💡 Sample Interaction
```
Ask your question: What are the top-rated books from the 2000s?

Answer:
Some of the top-rated books from the 2000s include
"The System of the World" by Neal Stephenson (avg rating 4.3, 16k ratings).
```

## 🔑 Key Components
- vector.py
    - Loads JSONL dataset
    - Normalizes schema (title, authors, year, ratings)
    - Creates row‑level and aggregate documents
    - Builds FAISS vector store with HuggingFace embeddings

- main.py
    - Initializes Ollama LLM (llama3.2)
    - Defines a prompt template combining retrieved docs + user question
    - Runs an interactive chat loop

## ✅ Summary
This project is a hands‑on showcase of Retrieval‑Augmented Generation (RAG) pipelines powered by Pandas, FAISS, LangChain, and Ollama. It emphasizes end‑to‑end skills in cleaning and structuring data, generating embeddings, performing efficient vector search, and orchestrating large language models for intelligent query answering.
