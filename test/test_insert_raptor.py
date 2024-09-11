import pandas as pd
import asyncio
from app.raptor.raptor_vectordb_insert import RAPTOR

async def test_raptor():
    raptor = RAPTOR()
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
    await raptor.load_data(data=data, metadata=metadata)

asyncio.run(test_raptor())