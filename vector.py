"""
Enhanced Book Embedding Pipeline with Pandas + FAISS + LangChain
-------------------------------------------------------
This script demonstrates how to:
1. Load and normalize book metadata from a JSONL file.
2. Create row-level and aggregate LangChain 'Document' objects.
3. Enrich documents with derived fields (decade, popularity, ratings).
4. Generate embeddings using HuggingFace models.
5. Store and retrieve documents efficiently with FAISS vector DB.

Author: Chandrakant Kokje
"""

import pandas as pd
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------------------------------------------
# Step 1: Load JSONL into a DataFrame
# -------------------------------------------------------------------
def load_jsonl_to_df(filePath: str, lines: bool = True, chunkSize: int = None) -> pd.DataFrame:
    """
    Load a JSONL file into a Pandas DataFrame with optional chunking.

    Args:
        filePath (str): Path to the JSONL file.
        lines (bool): Whether the file is line-delimited JSON (default: True).
        chunkSize (int): Number of rows per chunk for large files (default: None).

    Returns:
        pd.DataFrame: Normalized DataFrame with schema consistency enforced.
    """
    if chunkSize:
        dfs = []
        with pd.read_json(path_or_buf=filePath, lines=lines, chunksize=chunkSize) as reader:
            for chunk in reader:
                dfs.append(chunk)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_json(path_or_buf=filePath, lines=lines)

    # Enforce schema consistency for downstream processing
    df["title"] = df["title"].astype("string")
    df["authors"] = df["authors"].astype("object")  # keep as list
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce")

    # Derive decade for aggregation
    df["decade"] = (df["publication_year"] // 10) * 10

    return df.reset_index(drop=True)


# Load dataset
df = load_jsonl_to_df(filePath="books.jsonl", lines=True, chunkSize=10000)


# -------------------------------------------------------------------
# Step 2: Initialize Embeddings + Vector DB Config
# -------------------------------------------------------------------
db_location = "BooksDB"
model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

documents = []

# Normalize authors into separate rows for aggregation
df_exploded = df.explode("authors").copy()
df_exploded["authors"] = df_exploded["authors"].astype("string").str.strip()


# -------------------------------------------------------------------
# Step 3: Create LangChain Documents
# -------------------------------------------------------------------

# 3.1 Row-level documents (each book as a document)
for i, row in df.iterrows():
    doc = Document(
        page_content=(
            f"Book: {row['title']} by {', '.join(row['authors'])}. "
            f"Published in {row['publication_year']}, "
            f"average rating {row['average_rating']} from {row['ratings_count']} ratings."
        ),
        metadata={
            "id": row["id"],
            "title": row["title"],
            "authors": row["authors"],
            "year": int(row["publication_year"]) if pd.notnull(row["publication_year"]) else None,
            "decade": int(row["decade"]) if pd.notnull(row["decade"]) else None,
            "average_rating": float(row["average_rating"]) if pd.notnull(row["average_rating"]) else None,
            "ratings_count": int(row["ratings_count"]) if pd.notnull(row["ratings_count"]) else None,
            "image_url": row["image_url"],
            "type": "Book"
        },
        id=str(row["id"])
    )
    documents.append(doc)


# 3.2 Aggregate documents: Author-level summaries
books_by_author = (
    df_exploded.groupby("authors")
    .size()
    .reset_index(name="Count")
    .sort_values(by="Count", ascending=False)
)

for _, row in books_by_author.iterrows():
    aggDoc = Document(
        page_content=f"Author {row['authors']} has {row['Count']} books in the dataset.",
        metadata={"type": "AuthorAggregate", "authors": row["authors"], "Count": int(row["Count"])},
        id=f"Author-{row['authors']}"
    )
    documents.append(aggDoc)


# 3.3 Aggregate documents: Decade-level summaries
books_by_decade = (
    df.groupby("decade")
    .size()
    .reset_index(name="Count")
    .sort_values(by="decade", ascending=True)
)

for _, row in books_by_decade.iterrows():
    decade_doc = Document(
        page_content=f"In the {int(row['decade'])}s, {row['Count']} books were published.",
        metadata={"type": "DecadeAggregate", "decade": int(row["decade"]), "Count": int(row["Count"])},
        id=f"Decade-{row['decade']}"
    )
    documents.append(decade_doc)


# 3.4 Aggregate documents: Popularity signals (top-rated books)
top_books = df.sort_values(by="average_rating", ascending=False).head(50)
for _, row in top_books.iterrows():
    rating_doc = Document(
        page_content=(
            f"Highly rated book: {row['title']} by {', '.join(row['authors'])}, "
            f"average rating {row['average_rating']} from {row['ratings_count']} ratings."
        ),
        metadata={
            "type": "TopRated",
            "title": row["title"],
            "authors": row["authors"],
            "average_rating": float(row["average_rating"]),
            "ratings_count": int(row["ratings_count"])
        },
        id=f"TopRated-{row['id']}"
    )
    documents.append(rating_doc)


# -------------------------------------------------------------------
# Step 4: Build or Load FAISS Vector Store
# -------------------------------------------------------------------
if os.path.exists(db_location):
    vector_store = LangFAISS.load_local(
        folder_path=db_location,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Loaded existing FAISS vector store.")
else:
    vector_store = LangFAISS.from_documents(documents=documents, embedding=embeddings)
    vector_store.save_local(db_location)
    print("✅ Created and saved new FAISS vector store.")


# -------------------------------------------------------------------
# Step 5: Define Retriever
# -------------------------------------------------------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 50})

print("🚀 Retriever ready. Supports queries by book, author, decade, and popularity.")
