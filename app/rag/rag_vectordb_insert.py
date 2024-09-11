from typing import Dict, List
from langchain_community.document_loaders import NewsURLLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAG:
    def __init__(self, chunk_size=1000, overlap=200):
        from app.vectordb.vectordb import get_vectorstore
        self.chunk_size=chunk_size
        self.overlap=overlap
        self.vectordb = get_vectorstore("rag_collection")
    
    async def get_documents_by_news_url(self, url):
        try:
            # Document Load
            urls = []
            urls.append(url)
            loader = NewsURLLoader(
                urls=urls, 
                text_mode=True,
                nlp=True,
                show_progress_bar =True
            )
            documents = await loader.aload()
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            documents = None
        return documents
    
    async def split_chunk_text(self, text: str) -> List[str]:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
            split_texts = text_splitter.split_text(text)
        except Exception as e:
            print(f"Split texts failed : {e}")
        return split_texts

    async def insert_additional_info(self, text_list: list[str], metadata: Dict, meta_level: str) -> List[Document]:
        """
        Description: 
        text_list : news text list (leaf node)
        metadata : week, day, topic, news_index, "제목", "URL"
        """
        insert_docs = []
        for textIdx in range(len(text_list)):            
            text = text_list[textIdx]
            metadata[meta_level] = 0
            metadata["topic_level"]=0
            metadata["day_level"]=0
            insert_docs.append(Document(page_content=text, metadata=metadata))
        return insert_docs

    async def insert_vectordb(self, documents_list: List[Document]):
        try:
            if self.vectordb is not None and documents_list is not None:
                self.vectordb.add_documents(documents_list)
                print("vectordb add documents!!!")
            else:
                print("Vector DB or documents_list is None")
        except Exception as e:
            print(f"Vector DB insert failed : {e}")
            return 1

    async def load_data(self, data, metadata):
        """
        Description: 
        data : topic dataframe data
        metadata : week, day, topic
        """
        # extract url from data
        news_cnt = 0
        for idx, row in data.iterrows():
            url = row['URL']
            metadata["news_index"]= news_cnt + 1
            metadata["제목"] = row['제목']
            metadata["URL"] = url
            print(f"제목: {row['제목']}, url : {url}")
            if news_cnt >= 10:
                print(f"Extract news text finish !!! : news count : {news_cnt}")
                break
            if isinstance(url, str) and url.strip() != "" and url is not None:  
                try:
                    # Document Loader
                    documents= await self.get_documents_by_news_url(url)
                    if documents is not None:
                        for document in documents:
                            # Split Text
                            text = document.page_content 
                            text_list = await self.split_chunk_text(text)
                            insert_docs = await self.insert_additional_info(text_list=text_list, metadata=metadata, meta_level="news_level")
                            # Embedding
                            insert_result = await self.insert_vectordb(documents_list=insert_docs)
                            if insert_result == 1: # if success
                                news_cnt += 1   
                except Exception as e:
                    print(f"Failed to process URL {url}: {e}")
                    pass