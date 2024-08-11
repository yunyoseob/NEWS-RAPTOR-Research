from functools import lru_cache
import os

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from langchain_postgres.vectorstores import PGVector
from sqlalchemy import text
from langchain_openai import OpenAIEmbeddings

vectordb_HOST=os.getenv('vectordb_HOST')
vectordb_PORT=os.getenv('vectordb_PORT')
vectordb_DB=os.getenv('vectordb_DB')
vectordb_USER=os.getenv('vectordb_USER')
vectordb_PW=os.getenv('vectordb_PW')

@lru_cache(maxsize=128)
def get_vectorstore(collection_name: str) -> PGVector:
    connection_string =(
                    f"postgresql+psycopg://{vectordb_USER}:{vectordb_PW}@"
                    f"{vectordb_HOST}:{vectordb_PORT}/{vectordb_DB}"
                )

    try:    
        engine = create_engine(
            connection_string,
            pool_size=10,                    # 기본적으로 유지할 연결의 수를 설정합니다.
            max_overflow=20,                 # 기본 풀 크기를 초과하여 추가로 생성할 수 있는 연결의 수를 설정합니다.
            pool_timeout=30,                 # 연결 풀이 고갈되었을 때 새 연결을 얻기 위해 대기할 최대 시간을 설정합니다.
            pool_pre_ping=True,              # 연결이 유효한지 미리 확인하여 유효하지 않은 경우 새로운 연결을 생성합니다.
            pool_recycle=1800,               # 일정 시간(30분) 동안 사용되지 않은 연결을 자동으로 재활용합니다.
            pool_reset_on_return='rollback'  # 연결이 반환될 때 트랜잭션을 롤백하여 일관된 상태를 유지합니다.
        )
        vectorstore =  PGVector(
                        connection=engine,
                        embeddings= OpenAIEmbeddings(model="text-embedding-3-large"),
                        collection_name=collection_name
                    )
    except OperationalError as e:
        print(f"Database connection failed: {e}")
        raise

    return vectorstore