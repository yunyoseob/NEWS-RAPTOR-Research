import pandas as pd
import asyncio
from app.rag.rag_vectordb_insert import RAG

async def test_rag():
    rag = RAG()
    metadata = {}
    metadata["week"]="8월 5주"
    metadata["day"]="20240830"
    metadata["topic"]="국가 지급 보장' 법문화…청년세대 수긍할 개혁"
    data = pd.read_excel("국가 지급 보장' 법문화…청년세대 수긍할 개혁.xlsx")
    """
    Description: 
    data : topic dataframe data
    metadata : week, day, topic
    """
    # extract url from data
    news_cnt = 0
    documents_list = []
    for idx, row in data.iterrows():
        url = row['URL']
        metadata["news_index"]= news_cnt + 1
        metadata["제목"] = row['제목']
        metadata["뉴스 식별자"] = row['뉴스 식별자']
        metadata["URL"] = url
        if news_cnt >= 10:
            print(f"Extract news text finish !!! : news count : {news_cnt}")
            break
        if pd.notna(url):  # Check if url is not NaN
            try:
                documents= await rag.get_documents_by_news_url(url)
                if documents is not None:
                    for document in documents:
                        # Split Text
                        text = document.page_content 
                        text_list = await rag.split_chunk_text(text)
                        insert_docs = await rag.insert_additional_info(text_list=text_list, metadata=metadata, meta_level="news_level")
                        # text add
                        documents_list.extend(insert_docs)       
                news_cnt += 1
            except Exception as e:
                print(f"Failed to process URL {url}: {e}")
                pass
    print("=======================================")
    print(f"documents_list length : {len(documents_list)}")
    print(f"documents_list : {documents_list}")

asyncio.run(test_rag())