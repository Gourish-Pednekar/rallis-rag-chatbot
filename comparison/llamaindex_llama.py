"""LlamaIndex comparison pipeline using Gemini embeddings and an Ollama LLM."""

from dotenv import load_dotenv
import pdfplumber
import re
import os
import time
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.ollama import Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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

# ---------- STEP 1: Extract text ----------
print("Extracting text from catalogue...")
docs = []
with pdfplumber.open("catalogue.pdf") as pdf:
    for i, page in enumerate(pdf.pages):
        raw_text = page.extract_text(layout=True)
        if raw_text and raw_text.strip():
            # Keep page metadata so retrieved source nodes can be traced later.
            cleaned = clean_text(raw_text)
            docs.append(Document(
                text=cleaned,
                metadata={"page": i + 1}
            ))

print(f"Pages loaded: {len(docs)}")

# ---------- STEP 2: Configure embeddings and LLM ----------
# Configure LangChain Gemini embeddings for use inside LlamaIndex.
lc_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
embed_model = LangchainEmbedding(lc_embeddings)
llm = Ollama(model="llama3.2:1b", request_timeout=120.0)

# Apply global LlamaIndex settings for this comparison run.
Settings.embed_model = embed_model
Settings.llm = llm
Settings.chunk_size = 1200
Settings.chunk_overlap = 200

# ---------- STEP 3: Build index in batches ----------
print("Building LlamaIndex vector index in batches...")

batch_size = 20
index = None

for i in range(0, len(docs), batch_size):
    batch = docs[i:i + batch_size]
    batch_num = i // batch_size + 1
    total_batches = (len(docs) - 1) // batch_size + 1
    print(f"Processing batch {batch_num}/{total_batches} (pages {i+1} to {min(i+batch_size, len(docs))})")

    # Create the index from the first batch, then insert later documents.
    if index is None:
        index = VectorStoreIndex.from_documents(batch)
    else:
        for doc in batch:
            index.insert(doc)

    if i + batch_size < len(docs):
        time.sleep(15)

print("Index built.")

# ---------- STEP 4: Query ----------
# Build a query engine to test retrieval and answer generation.
query_engine = index.as_query_engine(similarity_top_k=3)

query = "What product should I use to treat stemborer in rice?"
print(f"\nQuery: {query}")
response = query_engine.query(query)
print(f"\nAnswer: {response}")
print("\nSource pages:")
for node in response.source_nodes:
    print(f"  - Page {node.metadata['page']}")
