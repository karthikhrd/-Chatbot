import streamlit as st


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GROQ_API_KEY   = st.secrets["GROQ_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
LLM_MODEL      = st.secrets.get("LLM_MODEL", "gpt-3.5-turbo")










# # Streamlit settings
# STREAMLIT_SECRET_KEY = os.getenv("STREAMLIT_SECRET_KEY", "")

# import os
# from dotenv import load_dotenv

# # Load .env file
# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

