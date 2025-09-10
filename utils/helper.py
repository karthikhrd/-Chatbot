from langchain_core.messages import HumanMessage, SystemMessage
from app import safe_invoke  # careful: avoid circular import

def process_user_message(chat_model, user_input, system_prompt=""):
    """Helper to process user input and return model response"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]
    return safe_invoke(chat_model, messages)

