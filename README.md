# RAG-Based Multi-PDF Chatbot for Farmers
Rallis India Limited - Internship Project

## Overview
A Retrieval-Augmented Generation (RAG) based chatbot that answers farmer queries 
based on the Rallis India product catalogue.

## Tech Stack
- LangChain + FAISS + Google Gemini Embeddings
- Groq API (Llama 3.1 8B) for response generation
- Streamlit for UI
- ngrok for deployment

## Setup
1. Clone the repository
2. Create a virtual environment: python -m venv venv
3. Activate: venv\Scripts\activate
4. Install dependencies: pip install -r requirements.txt
5. Copy .env.example to .env and add your API keys
6. Build the vector store: python build_vectorstore.py
7. Run the app: streamlit run app.py

## Project Structure
- app.py - Main Streamlit chatbot interface
- main_pipeline.py - Core RAG pipeline
- build_vectorstore.py - FAISS index builder
- comparison/ - LangChain vs LlamaIndex comparison scripts
- tests/ - Test and utility scripts
