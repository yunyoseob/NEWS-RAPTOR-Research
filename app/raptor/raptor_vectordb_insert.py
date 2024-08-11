from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

class RAPTOR:
    def __init__(self, chunk_size=1000, overlap=200):
        self.chunk_size=chunk_size
        self.overlap=overlap
        self.embed_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    async def split_chunk_text(self, text):
        splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        chunks = splitter.split_text(text)
        return chunks
    
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
                    documents= await self.get_documents_by_news_url(url)
                    if documents is not None:
                        for doc in documents:
                            text = doc.page_content
                            chunks = await self.split_chunk_text(text)
                            print(f"Chunks for URL {url}:")
                            for chunk in chunks:
                                print(chunk)
                                
                        news_cnt += 1
                except Exception as e:
                    print(f"Failed to process URL {url}: {e}")
                    pass