from app.vectordb.vectordb import get_vectorstore
from sqlalchemy import text
"""
topic summary = day leaf node
day summary = week leaf node
"""
def test_query(meta: str, meta_level: str):
    vectorstore = get_vectorstore("raptor_collection")
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
        print(result_dicts)
            
test_query("topic", "news_level")


