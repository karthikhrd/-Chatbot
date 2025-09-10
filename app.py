import streamlit as st
import os
import sys
import base64
import pandas as pd
from datetime import datetime
import PyPDF2
from dotenv import load_dotenv


load_dotenv()


# Read API keys from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")

# LangChain core
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Models
from models.llm import get_chatgroq_model  # Your helper for Groq/OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain for PDF Q&A
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# ---------------- Utility Functions ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_pdf_qa_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in context, reply: "Answer is not available in the context."
    
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def get_chat_response(chat_model, messages, system_prompt):
    """General chatbot response"""
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Error getting response: {str(e)}"


# ---------------- Pages ----------------
def instructions_page():
    st.title("📘 Instructions & Setup")
    st.markdown("""
    ## 🔧 Installation
    ```bash
    pip install -r requirements.txt
    ```
    ## API Keys
    - Stored in `.env` file (OPENAI, GROQ, GOOGLE)  
    
    ## Features
    - **Chat** with any LLM provider (Groq, OpenAI, Gemini)  
    - **Ask Questions from PDFs** (Document Q&A)  
    - **Download conversation history** as CSV  

    Ready? Go to the sidebar → and choose a page! 🚀
    """)


def chat_page():
    st.title("🤖 General ChatBot")

    # Use Gemini model by default
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chat_response(chat_model, st.session_state.messages, "")
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# def pdf_chat_page():
#     st.title("📄 Chat with Your PDFs")

#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []

#     pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
#     user_question = st.text_input("Ask a question about your PDFs")

#     if user_question and pdf_docs:
#         text_chunks = get_text_chunks(get_pdf_text(pdf_docs))
#         vector_store = get_vector_store(text_chunks)

#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
#         new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         docs = new_db.similarity_search(user_question)

#         chain = get_pdf_qa_chain()
#         response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

#         st.session_state.conversation_history.append(
#             (user_question, response["output_text"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#              ", ".join([pdf.name for pdf in pdf_docs]))
#         )

#     # Display history
#     for q, a, t, pdfs in reversed(st.session_state.conversation_history):
#         st.markdown(f"**You:** {q}\n\n**Bot:** {a}\n\n*({t} | PDFs: {pdfs})*")

#     # Download CSV
#     if len(st.session_state.conversation_history) > 0:
#         df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Timestamp", "PDFs"])
#         csv = df.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()
#         st.sidebar.markdown(
#             f'<a href="data:file/csv;base64,{b64}" download="pdf_chat_history.csv">📥 Download History</a>',
#             unsafe_allow_html=True
#         )
def pdf_chat_page():
    st.title("📄 Chat with Your PDFs")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
    user_question = st.text_input("Ask a question about your PDFs")

    if user_question and pdf_docs:
        # Process PDFs into chunks and vector store
        text_chunks = get_text_chunks(get_pdf_text(pdf_docs))
        vector_store = get_vector_store(text_chunks)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        # Run QA chain
        chain = get_pdf_qa_chain()
        response = chain(
            {"input_documents": docs, "query": user_question},   # ✅ fixed "question" → "query"
            return_only_outputs=True
        )

        # Handle missing/empty responses
        answer = response.get("output_text", "").strip()
        if not answer:
            answer = "⚠️ Sorry, I couldn’t generate a response. Try rephrasing your question."

        # Save to history
        st.session_state.conversation_history.append(
            (user_question, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             ", ".join([pdf.name for pdf in pdf_docs]))
        )

    # Display conversation history
    for q, a, t, pdfs in reversed(st.session_state.conversation_history):
        st.markdown(f"**You:** {q}\n\n**Bot:** {a}\n\n*({t} | PDFs: {pdfs})*")

    # Download CSV option
    if st.session_state.conversation_history:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Timestamp", "PDFs"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.sidebar.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="pdf_chat_history.csv">📥 Download History</a>',
            unsafe_allow_html=True
        )


# ---------------- Main ----------------
def main():
    st.set_page_config(
        page_title="Unified ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "PDF Chat", "Instructions"], index=0)

        if page == "Chat" and st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if page == "PDF Chat" and st.button("🗑️ Clear PDF History", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()

    if page == "Chat":
        chat_page()
    elif page == "PDF Chat":
        pdf_chat_page()
    else:
        instructions_page()


print(PyPDF2.__version__)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"🚨 App crashed: {e}")
        print(traceback.format_exc())

