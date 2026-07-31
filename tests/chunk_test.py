from dotenv import load_dotenv
import pdfplumber
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def clean_text(text):
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()

docs = []
with pdfplumber.open("catalogue.pdf") as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    for i, page in enumerate(pdf.pages):
        raw_text = page.extract_text(layout=True)
        if raw_text and raw_text.strip():
            cleaned = clean_text(raw_text)
            docs.append({
                "page": i + 1,
                "content": cleaned
            })

print(f"Pages with content: {len(docs)} (skipped {len(pdf.pages) - len(docs)} empty pages)")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)

all_chunks = []
for doc in docs:
    pieces = splitter.split_text(doc["content"])
    for piece in pieces:
        all_chunks.append({
            "page": doc["page"],
            "content": piece
        })

print(f"Total chunks created: {len(all_chunks)}")

single_chunk_pages = sum(1 for doc in docs if len(splitter.split_text(doc["content"])) == 1)
print(f"Pages that stayed as ONE chunk: {single_chunk_pages} out of {len(docs)}")

page3_chunks = [c for c in all_chunks if c["page"] == 3]
print(f"\nALSTOR (page 3) split into {len(page3_chunks)} chunk(s)")
for i, chunk in enumerate(page3_chunks):
    print(f"\n---- Chunk {i+1} ({len(chunk['content'])} chars) ----")
    print(chunk["content"])