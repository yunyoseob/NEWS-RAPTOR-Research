from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from app.vectordb.vectordb import VectorDB
import pandas as pd

class RAG:
    def __init__(self, chunk_size=1000, overlap=0):
        self.chunk_size=chunk_size
        self.overlap=overlap
        self.embed_model = OpenAIEmbeddings()
    
    async def split_chunk_text(self, text):
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        texts = text_splitter.split_text(text)
        return texts
    
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
    
    async def add_vectordb(self, texts):
        print(f"texts : {texts}")
        embeddings = []
        for text in texts:
            embedding = await self.embed_model.embed(text)
            embeddings.append(embedding)

        vectordb = VectorDB()
        rag_collection = vectordb.get_vectordb("rag_collection")
        # 벡터를 "rag_collection"에 삽입
        entities = [
            [i for i in range(len(embeddings))],  # ids
            embeddings  # embeddings
        ]
        #rag_collection.insert(entities)
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
                            text = document.page_content
                            # Split Text
                            texts = await self.split_chunk_text(text)
                            # Embedding
                            await self.add_vectordb(texts)
                    news_cnt += 1
                except Exception as e:
                    print(f"Failed to process URL {url}: {e}")
                    pass