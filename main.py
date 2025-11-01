"""
Interactive Book Q&A with LangChain + Ollama
--------------------------------------------
This script connects:
1. A FAISS retriever (vector.py) for semantic search over book data.
2. An Ollama LLM for natural language reasoning.
3. A simple chat loop for interactive Q&A.

Author: Chandrakant Kokje
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


# -------------------------------------------------------------------
# Step 1: Initialize the LLM
# -------------------------------------------------------------------
# Using Ollama with the llama3.2 model.
# `num_thread` controls parallelism for faster inference on multi-core CPUs.
model = OllamaLLM(model="llama3.2", num_thread=8)


# -------------------------------------------------------------------
# Step 2: Define Prompt Template
# -------------------------------------------------------------------
# The template combines retrieved context (reviews/metadata)
# with the user’s question to guide the LLM’s response.
template = """
You are an expert at answering questions about books.

Here are some relevant book entries:
{reviews}

Here is the user’s question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template=template)


# -------------------------------------------------------------------
# Step 3: Create Execution Chain
# -------------------------------------------------------------------
# The chain pipes the prompt into the model.
# This makes it easy to swap components (retriever, LLM, prompt) later.
chain = prompt | model


# -------------------------------------------------------------------
# Step 4: Interactive Chat Loop
# -------------------------------------------------------------------
# Continuously prompt the user for questions until they quit.
# Each question is enriched with retrieved book data before being answered.
print("📚 Book Q&A Assistant (type 'q' to quit)")
print("-----------------------------------------")

while True:
    question = input("\nAsk your question: ")
    if question.lower().strip() == "q":
        print("👋 Exiting Book Q&A. Goodbye!")
        break

    # Retrieve relevant book documents from FAISS vector store
    reviews = retriever.invoke(question)

    # Run the chain: inject retrieved docs + user question into the LLM
    result = chain.invoke({"reviews": reviews, "question": question})

    # Display the answer
    print("\n🔎 Answer:")
    print(result)
    print("--------------------------------------------------")
