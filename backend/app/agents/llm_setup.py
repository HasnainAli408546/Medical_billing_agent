from langchain_groq import ChatGroq
from app.core.config import settings

def get_llm():
    """
    Returns the configured LLM. We default to Groq's fast inference 
    endpoints utilizing LLaMA-3, as per the free API tier requirement.
    """
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
    
    # We use llama-3.3-70b-versatile for high-quality complex extraction/coding tasks.
    llm = ChatGroq(
        temperature=0,
        groq_api_key=settings.GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )
    return llm
