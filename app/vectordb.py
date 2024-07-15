from functools import lru_cache
import logging
from pymilvus import MilvusClient, DataType
import os

logger = logging.getLogger(__name__)

# 데이터베이스 파일의 절대 경로 생성
data_dir = os.path.join(os.path.dirname(__file__), '..', 'milvus', 'data')
db_path = os.path.join(data_dir, 'korean_news_vector.db')

# MilvusClient 생성
client = MilvusClient(db_path)
collection_name = "korea_news_collection"

def create_vectordb_collection():
    # 이미 존재하는지 확인 후 생성
    if not client.has_collection(collection_name):
        # 1. Create schema
        schema = client.create_schema(
            auto_id=False,  # 자동 ID 생성 비활성화
            enable_dynamic_field=False  # 동적 필드 활성화 비활성화
        )
        # 2. Add fields to schema
        schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
        
        # 3. Prepare index parameters
        index_params = client.prepare_index_params()

        # 4. Add indexes
        index_params.add_index(
            field_name="my_id",
            index_type="STL_SORT"
        )

        index_params.add_index(
            field_name="my_vector",
            index_type="AUTOINDEX",  # IVF_FLAT 인덱스 사용
            metric_type="L2",  # L2 거리 측정 방식
            params={"nlist": 1024}  # IVF_FLAT의 nlist 설정
        )

        # 5. Create a collection
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"Created collection: {collection_name}")
    else:
        logger.info(f"Collection already exists: {collection_name}")
    
@lru_cache(maxsize=None)  # Cache indefinitely
def get_vectordb():
    create_vectordb_collection()
    collection = client.get_collection(collection_name)
    return collection
    