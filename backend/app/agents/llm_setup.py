from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from app.core.config import settings

def get_llm():
    """
    Returns the configured LLM. We default to Groq's fast inference 
    endpoints utilizing LLaMA-3. If it hits rate limits, it falls back
    to OpenRouter.
    """
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")
    
    # 1. Primary Model (Groq - Free but strict rate limits)
    groq_llm = ChatGroq(
        temperature=0,
        groq_api_key=settings.GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )
    
    # If OpenRouter key isn't set, just return Groq as normal
    if not settings.OPENROUTER_API_KEY:
        return groq_llm
        
    # 2. Backup Model (OpenRouter - Paid/Flexible)
    openrouter_llm = ChatOpenAI(
        temperature=0,
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-3.3-70b-instruct",
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Billing Voice Agent",
        }
    )
    
    # 3. Combine them using LangChain's fallback mechanism
    robust_llm = groq_llm.with_fallbacks([openrouter_llm])
    
    return robust_llm
