from functools import lru_cache
from app.config import Settings, get_settings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

config = get_settings()

@lru_cache()
def get_chatllm_openai():
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.01,
        timeout=None,
        api_key=config.OPENAI_API_KEY
    )

@lru_cache
def get_openai_embeddings():
    embd = OpenAIEmbeddings(model="text-embedding-3-large")
    return embd