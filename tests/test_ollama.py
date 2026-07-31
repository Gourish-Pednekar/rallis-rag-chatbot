from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ---------- STEP 1: Load existing FAISS index ----------
print("Loading FAISS index...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("FAISS index loaded.")

# ---------- STEP 2: Set up Llama 3.2 via Ollama ----------
llm = OllamaLLM(model="llama3.2")

# ---------- STEP 3: Build prompt ----------
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant for farmers. Answer the question based only on the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
""")

# ---------- STEP 4: Build chain ----------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---------- STEP 5: Test ----------
query = "What product should I use to treat stemborer in rice?"
print(f"\nQuery: {query}")
answer = chain.invoke(query)
print(f"\nAnswer: {answer}")