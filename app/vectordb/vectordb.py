import os
from pymilvus import MilvusClient, DataType

class VectorDB:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'milvus', 'data')
        self.db_path = os.path.join(self.data_dir, 'korean_news_vector.db')
        self.client = MilvusClient(self.db_path) # MilvusClient 생성

    def create_vectordb_collection(self, collection_name):
        # 이미 존재하는지 확인 후 생성
        if not self.client.has_collection(collection_name):
            # 1. Create schema
            schema = self.client.create_schema(
                auto_id=False,  # 자동 ID 생성 비활성화
                enable_dynamic_field=False  # 동적 필드 활성화 비활성화
            )
            
            # 2. Add fields to schema
            schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
            
            # 3. Prepare index parameters
            index_params = self.client.prepare_index_params()

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
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection already exists: {collection_name}")
        
    def get_vectordb(self, collection_name):
        self.create_vectordb_collection(collection_name)
        collection = self.client.get_collection(collection_name)
        return collection
