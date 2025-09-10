# import os
# import sys
# from langchain_groq import ChatGroq
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# def get_chatgroq_model():
#     """Initialize and return the Groq chat model"""
#     try:
#         # Initialize the Groq chat model with the API key
#         groq_model = ChatGroq(
#             api_key="",
#             model="",
#         )
#         return groq_model
#     except Exception as e:
#         raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")

# models/llm.py
# import os
# import openai
# from config.config import OPENAI_API_KEY


# openai.api_key = OPENAI_API_KEY


# class OpenAIClient:
# def __init__(self, model: str = "gpt-3.5-turbo"):
# self.model = model


# def chat(self, messages, max_tokens=512, temperature=0.1):
# try:
# resp = openai.ChatCompletion.create(
# model=self.model,
# messages=messages,
# max_tokens=max_tokens,
# temperature=temperature,
# )
# return resp["choices"][0]["message"]["content"].strip()
# except Exception as e:
# raise RuntimeError(f"OpenAI request failed: {e}")

# models/llm.py
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY

from config.config import GROQ_API_KEY
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

def get_chatgroq_model(model_name="mixtral-8x7b-32768"):
    """Get Groq chat model, fallback to Gemini/OpenAI if no key"""
    if GROQ_API_KEY:
        return ChatGroq(model=model_name, api_key=GROQ_API_KEY)
    else:
        print("⚠️ No GROQ_API_KEY found, falling back to Gemini")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        # fallback to OpenAI if we want to  prefer:
        # here return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)




# def get_chatgroq_model(model_name="mixtral-8x7b-32768"):
#     if not GROQ_API_KEY:
#         raise ValueError("❌ GROQ_API_KEY is missing. Add it to your .env file.")
#     return ChatGroq(model=model_name, api_key=GROQ_API_KEY)

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def get_chat_model(provider: str = None, model_name: str = None):
    """
    Returns a chat model client based on provider and available API keys.
    Priority: user-specified -> environment detection
    """

    # Auto-detect provider if not given
    if not provider:
        if OPENAI_API_KEY:
            provider = "openai"
        elif GROQ_API_KEY:
            provider = "groq"
        elif GOOGLE_API_KEY:
            provider = "gemini"
        else:
            raise ValueError("❌ No API key found for OpenAI, Groq, or Google Gemini")

    # Default models per provider
    if provider == "openai":
        model_name = model_name or "gpt-3.5-turbo"
        return ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY)

    elif provider == "groq":
        model_name = model_name or "llama-3.1-8b-instant"
        return ChatGroq(model=model_name, api_key=GROQ_API_KEY)

    elif provider == "gemini":
        model_name = model_name or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY)

    else:
        raise ValueError(f"❌ Unknown provider: {provider}")


# Backwards-compatible Groq getter (for your old code)
def get_chatgroq_model():
    return get_chat_model(provider="groq")


