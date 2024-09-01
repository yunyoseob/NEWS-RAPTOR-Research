from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from app.vectordb.vectordb import get_vectorstore
import pandas as pd

class RAG:
    def __init__(self, chunk_size=1000, overlap=0):
        from app import get_openai_embeddings
        self.chunk_size=chunk_size
        self.overlap=overlap
        self.embed = get_openai_embeddings()
    
    async def split_chunk_text(self, document):
        try:
            text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
            split_document = text_splitter.split_documents([document])
        except Exception as e:
            print(f"Split document failed : {e}")
        return split_document
    
    async def get_documents_by_news_url(self, url):
        print(f"Processing URL: {url}")
        try:
            # Document Load
            loader = WebBaseLoader(url)
            documents = loader.load()
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            documents = None
        return documents
    
    async def add_vectordb(self, split_document):
        try:
            vectordb = get_vectorstore("rag_collection")
            vectordb.add_documents(split_document)
            print("vectordb add documents!!!")
        except Exception as e:
            print(f"Vector DB insert failed : {e}")
        return 1
        
    async def load_data(self, data):
        news_data_url = data['URL']
        print(news_data_url)
        print(f"news_data_url size : {news_data_url.size}")
        news_cnt = 0

        for idx, url  in enumerate(news_data_url):
            if news_cnt >= 10:
                print(f"Extract news text finish !!! : news count : {news_cnt}")
                break
            print(f"idx : {idx}, url : {url}")
            if pd.notna(url):  # Check if url is not NaN
                try:
                    # Document Loader
                    documents= await self.get_documents_by_news_url(url)
                    if documents is not None:
                        for document in documents:
                            # Split Text
                            split_document = await self.split_chunk_text(document)
                            # Embedding
                            await self.add_vectordb(split_document)
                    news_cnt += 1
                except Exception as e:
                    print(f"Failed to process URL {url}: {e}")
                    pass