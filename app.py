import streamlit as st
import base64
import pandas as pd
import time
from datetime import datetime
from PyPDF2 import PdfReader


import asyncio
import nest_asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()

# Project imports
from config.config import OPENAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, LLM_MODEL
from models.llm import get_chat_model

# ---------------- Streamlit Page Config ----------------
st.set_page_config(
    page_title="Unified ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Load Secrets ----------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
GROQ_API_KEY   = st.secrets.get("GROQ_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
LLM_MODEL      = st.secrets.get("LLM_MODEL", "gpt-3.5-turbo")


if not (OPENAI_API_KEY or GROQ_API_KEY or GOOGLE_API_KEY):
    st.error("⚠️ Missing API keys! Add them in `.streamlit/secrets.toml`")
    st.stop()

# ---------------- LangChain Imports ----------------
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ---------------- Provider Fallback ----------------
def get_available_chat_model(preferred_provider=None):
    """Return the first available chat model based on keys and fallback logic"""
    from models.llm import get_chat_model

    providers = []
    if preferred_provider:
        providers.append(preferred_provider)
    if OPENAI_API_KEY:
        providers.append("openai")
    if GROQ_API_KEY:
        providers.append("groq")
    if GOOGLE_API_KEY:
        providers.append("gemini")

    tried = []
    for provider in providers:
        try:
            chat_model = get_chat_model(provider=provider)
            _ = chat_model.invoke([HumanMessage(content="hello")])  # test
            return chat_model, provider
        except Exception as e:
            tried.append(f"{provider} → {str(e)}")
            continue

    st.error("⚠️ All providers failed! Tried:\n" + "\n".join(tried))
    st.stop()

# ---------------- Safe LLM Invocation ----------------
def safe_invoke(chat_model, messages):
    """Safely invoke chat_model.invoke() and handle common errors"""
    try:
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        err_str = str(e)
        if "insufficient_quota" in err_str:
            return "⚠️ Quota exceeded. Please check your plan or switch API key."
        elif "API key not valid" in err_str:
            return "⚠️ Invalid API key. Please check your key and provider."
        else:
            return f"⚠️ Error: {err_str}"

# ---------------- Streaming Simulation ----------------
def stream_response(text):
    """Simulate streaming effect for smoother UI"""
    placeholder = st.empty()
    output = ""
    for word in text.split():
        output += word + " "
        placeholder.markdown(
            f"""
            <div style='background-color:#F9F9F9; color:#222; 
                        padding:15px; border-radius:12px; margin:8px 0; 
                        font-size:16px; line-height:1.6; 
                        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);'>
                <b>Bot:</b> {output}▌
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.04)  # typing speed
    return output.strip()

# ---------------- Utility Functions ----------------
def get_chat_response(chat_model, messages, system_prompt=""):
    formatted_messages = [SystemMessage(content=system_prompt)] + [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in messages
    ]
    return safe_invoke(chat_model, formatted_messages)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    try:
        if not GOOGLE_API_KEY:
            st.error("⚠️ GOOGLE_API_KEY is missing! Add it to `.streamlit/secrets.toml`")
            st.stop()

        # Use smaller chunks for embedding stability
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return vector_store
    except Exception as e:
        st.error(f"⚠️ Embedding failed: {e}")
        return None


def get_pdf_qa_chain(chat_provider="gemini"):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in context, reply: "Answer is not available in the context."

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    try:
        model = get_chat_model(provider=chat_provider)
    except Exception as e:
        st.error(f"⚠️ Could not initialize PDF QA model: {e}")
        st.stop()

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# ---------------- Pages ----------------
def chat_page(chat_provider):
    st.markdown("<h1 style='text-align:center; color:#FF6F61;'>🤖 General ChatBot</h1>", unsafe_allow_html=True)

    try:
        chat_model, active_provider = get_available_chat_model(preferred_provider=chat_provider)
        st.info(f"Using provider: {active_provider}")
    except Exception as e:
        st.error(f"⚠️ Could not initialize any chat model: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "Bot"
        st.markdown(
            f"""
            <div style='background-color:{"#008000" if msg["role"]=="user" else "#F9F9F9"}; 
                        color:{"white" if msg["role"]=="user" else "#222"}; 
                        padding:12px; border-radius:10px; margin:8px 0; 
                        font-size:16px; line-height:1.6;'>
                <b>{role}:</b> {msg['content']}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chat_response(chat_model, st.session_state.messages)
                streamed = stream_response(response)  # typing effect
        st.session_state.messages.append({"role": "assistant", "content": streamed})
        st.experimental_rerun()

def pdf_chat_page(chat_provider):
    st.markdown("<h1 style='text-align:center; color:#6A5ACD;'>📄 Chat with Your PDFs</h1>", unsafe_allow_html=True)

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    col1, col2 = st.columns([3, 2])
    with col1:
        pdf_docs = st.file_uploader("📂 Upload PDF Files", accept_multiple_files=True, type=["pdf"])
    with col2:
        user_question = st.text_input("💬 Ask a question about your PDFs")

    if pdf_docs and user_question:
        with st.spinner("🔎 Processing documents..."):
            text_chunks = get_text_chunks(get_pdf_text(pdf_docs))
            get_vector_store(text_chunks)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)

        try:
            chat_model, active_provider = get_available_chat_model(preferred_provider=chat_provider)
            chain = get_pdf_qa_chain(chat_provider=active_provider)
        except Exception as e:
            st.error(f"⚠️ Could not initialize PDF QA model: {e}")
            return

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = chain({"input_documents": docs, "query": user_question}, return_only_outputs=True)
                answer = response.get("output_text", "⚠️ Sorry, no response.").strip()

        st.session_state.conversation_history.append(
            (user_question, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             ", ".join([pdf.name for pdf in pdf_docs]))
        )

    # Display styled Q&A history
    for q, a, t, pdfs in reversed(st.session_state.conversation_history):
        st.markdown(
            f"""
            <div style='background-color:#F3F6FF; padding:14px; border-radius:12px; margin:8px 0;
                        box-shadow: 0px 4px 12px rgba(0,0,0,0.1); animation: fadeInUp 0.4s;'>
                <b>📌 Question:</b> {q}<br>
                <b>🤖 Answer:</b> {a}<br>
                <i>🕒 {t} | 📂 PDFs: {pdfs}</i>
            </div>
            <style>
                @keyframes fadeInUp {{
                    from {{ opacity: 0; transform: translateY(10px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

    # Export history
    if st.session_state.conversation_history:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Timestamp", "PDFs"]
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.sidebar.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="pdf_chat_history.csv">📥 Download History</a>',
            unsafe_allow_html=True
        )

def instructions_page():
    st.markdown("<h1 style='text-align:center; color:#4682B4;'>📖 Instructions</h1>", unsafe_allow_html=True)
    st.write("""
    Welcome to the ChatBot Project!  

    - **Chat** → Talk to the LLM directly.  
    - **PDF Chat** → Upload PDFs and ask questions.  
    - **Provider** → Switch between OpenAI, Groq, Gemini.  
    - **Clear History** → Reset your conversation.  

    ⚡ Note: Ensure API keys are set in `.streamlit/secrets.toml`.
    """)

# ---------------- Main ----------------
def main():
    with st.sidebar:
        st.markdown("<h2 style='text-align:center;'>🧭 Navigation</h2>", unsafe_allow_html=True)

        page = st.radio("", ["Chat", "PDF Chat", "Instructions"], index=0)

        provider_options = []
        if OPENAI_API_KEY: provider_options.append("openai")
        if GROQ_API_KEY: provider_options.append("groq")
        if GOOGLE_API_KEY: provider_options.append("gemini")

        if not provider_options:
            st.error("⚠️ No valid API keys found. Please add them to `.streamlit/secrets.toml`")
            st.stop()

        chat_provider = st.selectbox("Choose LLM Provider", provider_options, index=0)

        if page == "Chat" and st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()
        if page == "PDF Chat" and st.button("🗑️ Clear PDF History", use_container_width=True):
            st.session_state.conversation_history = []
            st.experimental_rerun()

    if page == "Chat":
        chat_page(chat_provider)
    elif page == "PDF Chat":
        pdf_chat_page(chat_provider)
    else:
        instructions_page()

# ---------------- Run App ----------------
if __name__ == "__main__":
    main()



# import streamlit as st
# import base64
# import pandas as pd
# from datetime import datetime
# from PyPDF2 import PdfReader

# # Project imports
# from config.config import OPENAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, LLM_MODEL
# from models.llm import get_chat_model

# # ---------------- Streamlit Page Config ----------------
# st.set_page_config(
#     page_title="Unified ChatBot",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ---------------- Load Secrets ----------------
# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
# GROQ_API_KEY   = st.secrets.get("GROQ_API_KEY")
# GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
# LLM_MODEL      = st.secrets.get("LLM_MODEL", "gpt-3.5-turbo")

# if not (OPENAI_API_KEY or GROQ_API_KEY or GOOGLE_API_KEY):
#     st.error("⚠️ Missing API keys! Add them in `.streamlit/secrets.toml`")
#     st.stop()

# # ---------------- LangChain Imports ----------------
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate

# # ---------------- Provider Fallback ----------------
# def get_available_chat_model(preferred_provider=None):
#     """Return the first available chat model based on keys and fallback logic"""
#     from models.llm import get_chat_model

#     providers = []
#     if preferred_provider:
#         providers.append(preferred_provider)
#     if OPENAI_API_KEY:
#         providers.append("openai")
#     if GROQ_API_KEY:
#         providers.append("groq")
#     if GOOGLE_API_KEY:
#         providers.append("gemini")

#     tried = []
#     for provider in providers:
#         try:
#             chat_model = get_chat_model(provider=provider)
#             # quick test call (Gemini prefers HumanMessage)
#             _ = chat_model.invoke([HumanMessage(content="hello")])
#             return chat_model, provider
#         except Exception as e:
#             tried.append(f"{provider} → {str(e)}")
#             continue

#     st.error("⚠️ All providers failed! Tried:\n" + "\n".join(tried))
#     st.stop()

# # ---------------- Safe LLM Invocation ----------------
# def safe_invoke(chat_model, messages):
#     """Safely invoke chat_model.invoke() and handle common errors"""
#     try:
#         response = chat_model.invoke(messages)
#         return response.content
#     except Exception as e:
#         err_str = str(e)
#         if "insufficient_quota" in err_str:
#             return "⚠️ Quota exceeded. Please check your plan or switch API key."
#         elif "API key not valid" in err_str:
#             return "⚠️ Invalid API key. Please check your key and provider."
#         else:
#             return f"⚠️ Error: {err_str}"

# # ---------------- Utility Functions ----------------
# def get_chat_response(chat_model, messages, system_prompt=""):
#     formatted_messages = [SystemMessage(content=system_prompt)] + [
#         HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
#         for m in messages
#     ]
#     return safe_invoke(chat_model, formatted_messages)

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             text += page.extract_text() or ""
#     return text

# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     return splitter.split_text(text)

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")
#     return vector_store

# def get_pdf_qa_chain(chat_provider="gemini"):
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context. 
#     If the answer is not in context, reply: "Answer is not available in the context."

#     Context:\n {context}\n
#     Question: \n{question}\n

#     Answer:
#     """
#     try:
#         model = get_chat_model(provider=chat_provider)
#     except Exception as e:
#         st.error(f"⚠️ Could not initialize PDF QA model: {e}")
#         st.stop()

#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # ---------------- Pages ----------------
# def chat_page(chat_provider):
#     st.markdown("<h1 style='text-align:center; color:#FF6F61;'>🤖 General ChatBot</h1>", unsafe_allow_html=True)

#     try:
#         chat_model, active_provider = get_available_chat_model(preferred_provider=chat_provider)
#         st.info(f"Using provider: {active_provider}")
#     except Exception as e:
#         st.error(f"⚠️ Could not initialize any chat model: {e}")
#         st.stop()

#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display chat history with styled bubbles
#     for msg in st.session_state.messages:
#         role = "You" if msg["role"] == "user" else "Bot"
#         if msg["role"] == "user":
#             st.markdown(
#                 f"""
#                 <div style='background-color:#008000; color:white; 
#                             padding:12px; border-radius:10px; margin:8px 0; 
#                             font-weight:bold; animation: fadeIn 0.4s;'>
#                     <b>{role}:</b> {msg['content']}
#                 </div>
#                 <style>
#                     @keyframes fadeIn {{
#                         from {{ opacity: 0; transform: translateY(10px); }}
#                         to {{ opacity: 1; transform: translateY(0); }}
#                     }}
#                 </style>
#                 """,
#                 unsafe_allow_html=True
#             )
#         else:
#             if "⚠️" in msg["content"] or "Error" in msg["content"]:
#                 st.markdown(
#                     f"""
#                     <div style='background-color:#2E2E2E; color:#FF4C4C; 
#                                 padding:12px; border-radius:10px; margin:8px 0; 
#                                 font-weight:bold; animation: pulse 1s infinite; 
#                                 box-shadow: 0px 4px 10px rgba(255,0,0,0.5);'>
#                         <b>{role}:</b> {msg['content']}
#                     </div>
#                     <style>
#                         @keyframes pulse {{
#                             0% {{ box-shadow: 0 0 5px red; }}
#                             50% {{ box-shadow: 0 0 20px red; }}
#                             100% {{ box-shadow: 0 0 5px red; }}
#                         }}
#                     </style>
#                     """,
#                     unsafe_allow_html=True
#                 )
#             else:
#                 st.markdown(
#                     f"""
#                     <div style='background-color:#F9F9F9; color:#222; 
#                                 padding:15px; border-radius:12px; margin:8px 0; 
#                                 font-size:16px; line-height:1.6; 
#                                 box-shadow: 0px 4px 12px rgba(0,0,0,0.1); 
#                                 animation: fadeInUp 0.5s;'>
#                         <b>{role}:</b> {msg['content']}
#                     </div>
#                     <style>
#                         @keyframes fadeInUp {{
#                             from {{ opacity: 0; transform: translateY(15px); }}
#                             to {{ opacity: 1; transform: translateY(0); }}
#                         }}
#                     </style>
#                     """,
#                     unsafe_allow_html=True
#                 )

#     # Chat input
#     if prompt := st.chat_input("Type your message here..."):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = get_chat_response(chat_model, st.session_state.messages)
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         st.experimental_rerun()

# def pdf_chat_page(chat_provider):
#     st.markdown("<h1 style='text-align:center; color:#6A5ACD;'>📄 Chat with Your PDFs</h1>", unsafe_allow_html=True)

#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []

#     col1, col2 = st.columns([3, 2])
#     with col1:
#         pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
#     with col2:
#         user_question = st.text_input("Ask a question about your PDFs")

#     if pdf_docs and user_question:
#         text_chunks = get_text_chunks(get_pdf_text(pdf_docs))
#         get_vector_store(text_chunks)
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
#         new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         docs = new_db.similarity_search(user_question)
#         chat_model, active_provider = get_available_chat_model(preferred_provider=chat_provider)
#         chain = get_pdf_qa_chain(chat_provider=active_provider)
#         response = chain({"input_documents": docs, "query": user_question}, return_only_outputs=True)
#         answer = response.get("output_text", "⚠️ Sorry, no response.").strip()
#         st.session_state.conversation_history.append(
#             (user_question, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#              ", ".join([pdf.name for pdf in pdf_docs]))
#         )

#     for q, a, t, pdfs in reversed(st.session_state.conversation_history):
#         st.markdown(
#             f"<div style='background-color:#E8F0FE; padding:12px; border-radius:10px; margin:8px 0; "
#             f"box-shadow:0px 4px 12px rgba(0,0,0,0.1);'>"
#             f"<b>You:</b> {q}<br><b>Bot:</b> {a}<br><i>{t} | PDFs: {pdfs}</i></div>",
#             unsafe_allow_html=True
#         )

#     if st.session_state.conversation_history:
#         df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Timestamp", "PDFs"])
#         csv = df.to_csv(index=False)
#         b64 = base64.b64encode(csv.encode()).decode()
#         st.sidebar.markdown(
#             f'<a href="data:file/csv;base64,{b64}" download="pdf_chat_history.csv">📥 Download History</a>',
#             unsafe_allow_html=True
#         )

# def instructions_page():
#     st.markdown("<h1 style='text-align:center; color:#4682B4;'>📖 Instructions</h1>", unsafe_allow_html=True)
#     st.write("""
#     Welcome to the ChatBot Project!  

#     - **Chat** → Talk to the LLM directly.  
#     - **PDF Chat** → Upload PDFs and ask questions.  
#     - **Provider** → Switch between OpenAI, Groq, Gemini.  
#     - **Clear History** → Reset your conversation.  

#     ⚡ Note: Ensure API keys are set in `.streamlit/secrets.toml`.
#     """)

# # ---------------- Main ----------------
# def main():
#     with st.sidebar:
#         st.markdown("<h2 style='text-align:center;'>🧭 Navigation</h2>", unsafe_allow_html=True)

#         page = st.radio("", ["Chat", "PDF Chat", "Instructions"], index=0)

#         provider_options = []
#         if OPENAI_API_KEY: provider_options.append("openai")
#         if GROQ_API_KEY: provider_options.append("groq")
#         if GOOGLE_API_KEY: provider_options.append("gemini")

#         if not provider_options:
#             st.error("⚠️ No valid API keys found. Please add them to `.streamlit/secrets.toml`")
#             st.stop()

#         chat_provider = st.selectbox("Choose LLM Provider", provider_options, index=0)

#         if page == "Chat" and st.button("🗑️ Clear Chat History", use_container_width=True):
#             st.session_state.messages = []
#             st.experimental_rerun()
#         if page == "PDF Chat" and st.button("🗑️ Clear PDF History", use_container_width=True):
#             st.session_state.conversation_history = []
#             st.experimental_rerun()

#     if page == "Chat":
#         chat_page(chat_provider)
#     elif page == "PDF Chat":
#         pdf_chat_page(chat_provider)
#     else:
#         instructions_page()

# # ---------------- Run App ----------------
# if __name__ == "__main__":
#     main()
