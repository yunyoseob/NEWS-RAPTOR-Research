import os
import pandas as pd
import asyncio
from tqdm import tqdm
from app.rag.rag_vectordb_insert import RAG
from app.config import get_settings

config = get_settings()
date_list = ["20240826","20240827","20240828","20240829","20240830"]

# 날짜별 디렉토리 빈환
async def get_excel_file_dir_list():
    data_dir = os.path.join(config.PROJECT_ROOT_DIR, 'data')
    excel_file_dir_list=[]
    for date in date_list:
        excel_file_dir = data_dir + "/" + date
        excel_file_dir_list.append(excel_file_dir)
    return excel_file_dir_list

# 특정 날짜에 해당하는 10개의 이슈들에 대한 내용이 담긴 파일 리스트 반환
async def get_file_list(excel_file_dir: str) -> list[str]:
    file_list = []
    absolute_path = os.path.abspath(excel_file_dir)
    if os.path.exists(absolute_path):
        file_list = os.listdir(absolute_path)
    else:
        file_list = ["Directory not found"]
    return file_list

# RAG Collect Process Start
async def news_collect_start():
    rag = RAG(chunk_size=1000, overlap=200)
    excel_file_dir_list = await get_excel_file_dir_list()
    metadata = {}
    metadata["week"]="8월 5주"

    # 주간 이슈 내의 날짜별 디렉토리를 가져와서 RAG 구성
    for idx, excel_file_dir in enumerate(tqdm(excel_file_dir_list, desc="Start Read Weekly News By RAG")):        
        news_day = date_list[idx]
        metadata["day"]=news_day

        daily_topic_list = await get_file_list(excel_file_dir)

        # NEWS RAG: 주간 이슈의 특정 일자의 10개의 이슈 중 각 이슈(topic)에 대한 기사들로 RAG를 구성
        for file_name in tqdm(daily_topic_list, desc="Start Read Daily News"):
            metadata["topic"]=file_name

            file_path = excel_file_dir + "/" + file_name
            data = pd.read_excel(file_path)
            await rag.load_data(data=data, metadata=metadata)

asyncio.run(news_collect_start())