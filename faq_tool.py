import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH")

def clean_text(text):
    """Remove caracteres de formatação markdown e outros indesejados do texto."""
    import re
    
    # Remove marcações de negrito e itálico com asteriscos
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **texto**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove *texto*
    
    # Remove marcadores de lista com asterisco
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)  # Remove * no início das linhas
    text = re.sub(r'\n\s*\*\s+', '\n', text)  # Remove * depois de quebras de linha
    
    # Remove quaisquer asteriscos restantes
    text = text.replace('*', '')
    
    # Limpa espaços múltiplos e quebras de linha
    text = re.sub(r'\s+', ' ', text)  # Substitui múltiplos espaços por um
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Remove múltiplas quebras de linha
    
    return text.strip()

def get_faq_context(question):
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Limpa o texto de cada chunk antes de criar embeddings
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)
    db = FAISS.from_documents(chunks, embeddings)
    results = db.similarity_search(question, k=6)

    return results