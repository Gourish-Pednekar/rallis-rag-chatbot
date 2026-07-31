"""Core LangChain pipeline for the Rallis India Limited RAG chatbot."""

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def load_vectorstore():
    """Load the persisted FAISS vector store with Gemini embeddings.

    Parameters:
        None.

    Returns:
        A FAISS vector store loaded from the local faiss_index directory.
    """
    # Recreate the same embedding model used when the index was built.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def build_chain(vectorstore, temperature=0.3, system_role="You are a helpful assistant for farmers.", language="English"):
    """Build the retrieval and generation chain for catalogue question answering.

    Parameters:
        vectorstore: FAISS vector store containing embedded catalogue chunks.
        temperature: Sampling temperature for the Groq chat model.
        system_role: Role instruction used to shape the assistant response.
        language: Language in which the response should be generated.

    Returns:
        A tuple containing the runnable chain and its retriever.
    """
    # Retrieve the most relevant catalogue chunks for each question.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Configure the chat model used to generate grounded responses.
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)

    # Define the prompt rules that constrain answers to catalogue context.
    prompt = ChatPromptTemplate.from_template("""
{system_role}
The Rallis catalogue contains 156 products across insecticides, fungicides, herbicides, and fertilizers.

STRICT RULES:
- ONLY use information from the context below. Do not use any outside knowledge.
- If the answer is not in the context, say exactly: "I don't have that information in the Rallis catalogue."
- Never guess or make up product names, dosages, or ingredients.
- Always mention: product name, active ingredient, recommended dose, and time of application if available.
- IMPORTANT: Always respond in {language} language only.

Context:
{context}

Question: {question}

Answer strictly based on the context above:
""")

    def format_docs(docs):
        """Combine retrieved documents into prompt-ready context text.

        Parameters:
            docs: Retrieved LangChain Document objects.

        Returns:
            A string containing the joined page content for the prompt context.
        """
        return "\n\n".join(doc.page_content for doc in docs)

    # Compose retrieval, prompting, model inference, and output parsing.
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt.partial(system_role=system_role, language=language)
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

def get_answer(chain, retriever, query):
    """Generate an answer and collect source page references.

    Parameters:
        chain: Runnable RAG chain used to answer the query.
        retriever: Retriever used to fetch source documents for attribution.
        query: User question about Rallis catalogue content.

    Returns:
        A tuple containing the generated answer and a list of source page numbers.
    """
    answer = chain.invoke(query)
    source_docs = retriever.invoke(query)
    source_pages = list(set([doc.metadata["page"] for doc in source_docs]))
    return answer, source_pages

# ---------- Test ----------
if __name__ == "__main__":
    print("Loading vectorstore...")
    vectorstore = load_vectorstore()
    chain, retriever = build_chain(vectorstore)
    print("Ready.")

    query = "What product should I use to treat stemborer in rice?"
    print(f"\nQuery: {query}")
    answer, pages = get_answer(chain, retriever, query)
    print(f"\nAnswer: {answer}")
    print(f"\nSource pages: {pages}")


