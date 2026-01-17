import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

class RAGEngine:
    def __init__(self, api_key):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        self.vector_store = None

    def process_pdf(self, pdf_file):
        """Extracts text from PDF and builds FAISS index."""
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        # Split text for granular retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = text_splitter.split_text(text)
        
        # Create Vector Store
        self.vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
        return text, len(chunks)

    def get_context(self, query, k=3):
        """Retrieves relevant chunks for a specific concept."""
        if not self.vector_store:
            return ""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n".join([d.page_content for d in docs])
