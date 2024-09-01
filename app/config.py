from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

class Settings(BaseSettings):
    # openai key
    OPENAI_API_KEY: str

    # project directory
    PROJECT_ROOT_DIR: str

    # pgvector
    vectordb_HOST: str
    vectordb_PORT: str
    vectordb_DB: str
    vectordb_USER: str
    vectordb_PW: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

    def __hash__(self):
        return hash((self.app_name, ...))
    
@lru_cache
def get_settings():
    load_dotenv() 
    return Settings()