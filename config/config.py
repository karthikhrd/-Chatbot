# import os
# import streamlit as st
# from dotenv import load_dotenv

# # Try loading from .env (for local dev)
# load_dotenv()

# # First check Streamlit secrets, else fallback to env vars
# GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
# LLM_MODEL = st.secrets.get("LLM_MODEL", os.getenv("LLM_MODEL", "gpt-3.5-turbo"))























# config/config.py
# Keep API keys out of source control. Load from environment variables.
# import os
# from dotenv import load_dotenv


# load_dotenv()


# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY") # optional
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
# CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


# # Streamlit settings
# STREAMLIT_SECRET_KEY = os.getenv("STREAMLIT_SECRET_KEY", "")

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

