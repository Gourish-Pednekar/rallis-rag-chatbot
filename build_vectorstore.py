"""Build the FAISS vector store from the Rallis India Limited catalogue PDF."""

from dotenv import load_dotenv
import pdfplumber
import re
import os
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()


def clean_text(text):
    """Normalize text extracted from the product catalogue.

    Parameters:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text with compact spacing and stripped outer whitespace.
    """
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()

# ---------- STEP 1: Extract and clean text ----------
docs = []
with pdfplumber.open("catalogue.pdf") as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    for i, page in enumerate(pdf.pages):
        raw_text = page.extract_text(layout=True)
        if raw_text and raw_text.strip():
            # Preserve page numbers for source attribution during retrieval.
            cleaned = clean_text(raw_text)
            docs.append({"page": i + 1, "content": cleaned})

print(f"Pages with content: {len(docs)}")

# ---------- STEP 2: Chunk ----------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

langchain_docs = []
for doc in docs:
    pieces = splitter.split_text(doc["content"])
    for piece in pieces:
        # Store each chunk as a LangChain document with its source page.
        langchain_docs.append(
            Document(
                page_content=piece,
                metadata={"page": doc["page"]}
            )
        )

print(f"Total chunks created: {len(langchain_docs)}")

# ---------- STEP 3: Generate embeddings + build FAISS index (batched) ----------
print("Generating embeddings in batches (to respect free tier rate limits)...")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

batch_size = 20
vectorstore = None

for i in range(0, len(langchain_docs), batch_size):
    batch = langchain_docs[i:i + batch_size]
    batch_num = i // batch_size + 1
    total_batches = (len(langchain_docs) - 1) // batch_size + 1
    print(f"Processing batch {batch_num}/{total_batches} (chunks {i+1} to {min(i+batch_size, len(langchain_docs))})")

    # Create the index from the first batch, then append later batches.
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)

    if i + batch_size < len(langchain_docs):  # don't sleep after the last batch
        time.sleep(10)

print("All embeddings generated.")

# ---------- STEP 4: Save FAISS index to disk ----------
vectorstore.save_local("faiss_index")
print("FAISS index saved to 'faiss_index/' folder")

# ---------- STEP 5: Test retrieval ----------
test_query = "what treats stemborer in rice"
results = vectorstore.similarity_search(test_query, k=2)

print(f"\n---- Test Query: '{test_query}' ----")
for i, r in enumerate(results):
    print(f"\nResult {i+1} (from page {r.metadata['page']}):")
    print(r.page_content[:300])
