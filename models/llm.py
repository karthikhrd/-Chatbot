import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load secrets safely
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
GROQ_API_KEY   = st.secrets.get("GROQ_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
LLM_MODEL      = st.secrets.get("LLM_MODEL", "gpt-3.5-turbo")


def get_chat_model(provider="openai"):
    """
    Return a chat model instance for the given provider.
    Supported: openai, groq, gemini
    """

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY in secrets.toml")
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model=LLM_MODEL,  # e.g., gpt-3.5-turbo or gpt-4
            temperature=0.7
        )

    elif provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY in secrets.toml")
        # ✅ Updated Groq model (old Mixtral was deprecated)
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",   # use Groq docs for other models
            temperature=0.7
        )

    elif provider == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("Missing GOOGLE_API_KEY in secrets.toml")
        return ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model="gemini-1.5-flash",   # fast + cheaper than Pro
            temperature=0.7
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")
