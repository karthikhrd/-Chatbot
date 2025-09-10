from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.embeddings import get_embeddings
from models.llm import get_llm
from langchain.chains import RetrievalQA
from utils.web_utils import simple_web_search

def build_vectorstore(file_path: str):
    """Build a FAISS vectorstore from a PDF"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = get_embeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)

def get_rag_answer(vectorstore, query: str) -> str:
    """Retrieve answer using RAG pipeline with web fallback"""
    retriever = vectorstore.as_retriever()
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa.run(query)

    # If answer is too short or generic, fallback to web search
    if not answer or len(answer.strip()) < 10 or "I don't know" in answer:
        web_result = simple_web_search(query)
        return f"(Fallback Web Search) {web_result}"
    
    return answer
