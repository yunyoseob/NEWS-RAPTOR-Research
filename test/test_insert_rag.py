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
    await rag.load_data(data=data, metadata=metadata)

asyncio.run(test_rag())