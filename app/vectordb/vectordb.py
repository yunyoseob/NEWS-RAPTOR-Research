from functools import lru_cache
import os

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from langchain_postgres.vectorstores import PGVector
from sqlalchemy import text
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings

config = get_settings()

@lru_cache(maxsize=128)
def get_vectorstore(collection_name: str) -> PGVector:
    connection_string =(
                    f"postgresql+psycopg://{config.vectordb_USER}:{config.vectordb_PW}@"
                    f"{config.vectordb_HOST}:{config.vectordb_PORT}/{config.vectordb_DB}"
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

def get_summary_info(meta: str, meta_level: str):
    """
    Input
    meta: topic, day 등 조회하고 싶은 정보 범위
    meta_level: 조회하고 싶은 정보 범위를 찾기 위해 하위 메타 레벨 정보 제공 (news_level, topic_level ...)
    
    Output
    meta: topic, day 등 조회하고 싶은 정보 범위
    document: 조회하고 싶은 정보 범위에 해당하는 최상단 노드(ROOT NODE)의 텍스트
    max_meta_level: 조회하고 싶은 정보 범위에 해당하는 최상단 노드(ROOT NODE)의 레벨
    """
    vectorstore = get_vectorstore("raptor_collection")
    result_dicts = []
    query = text(f"""
            WITH RankedDocuments AS (
                SELECT cmetadata->>'{meta}' AS meta
                     , document
                     , (cmetadata->>'{meta_level}')::INT AS meta_level
                     , ROW_NUMBER() OVER (PARTITION BY cmetadata->>'{meta}' ORDER BY (cmetadata->>'{meta_level}')::INT DESC) AS rank
                  FROM langchain_pg_embedding
                 WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = 'raptor_collection')  
            )
            SELECT meta
                 , document
                 , meta_level AS max_meta_level
              FROM RankedDocuments
             WHERE 1=1
               AND rank = 1 
            ORDER BY max_meta_level DESC
            """)
    with vectorstore._session_maker() as session:
        key_list = ["meta", "document", "max_meta_level"]
        result = session.execute(query)
        rows = result.fetchall()
        result_dicts = [dict(zip(key_list, row)) for row in rows]
    return result_dicts