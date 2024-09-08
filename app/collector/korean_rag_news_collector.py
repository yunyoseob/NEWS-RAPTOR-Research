import os
import pandas as pd
import asyncio
from tqdm import tqdm
from app.rag.rag_vectordb_insert import RAG
from app.config import get_settings

config = get_settings()
date_list = ["20240826","20240827","20240828","20240829","20240830"]

async def get_excel_file_dir_list():
    data_dir = os.path.join(config.PROJECT_ROOT_DIR, 'data')
    excel_file_dir_list=[]
    for date in date_list:
        excel_file_dir = data_dir + "/" + date
        excel_file_dir_list.append(excel_file_dir)
    return excel_file_dir_list

async def get_file_list(excel_file_dir: str) -> list[str]:
    file_list = []
    absolute_path = os.path.abspath(excel_file_dir)
    if os.path.exists(absolute_path):
        file_list = os.listdir(absolute_path)
    else:
        file_list = ["Directory not found"]
    return file_list

async def news_collect_start():
    rag = RAG(chunk_size=1000, overlap=200)
    excel_file_dir_list = await get_excel_file_dir_list()
    print(f"excel_file_dir_list : {excel_file_dir_list}")

    # 1. 날짜 디렉토리
    for idx, excel_file_dir in enumerate(tqdm(excel_file_dir_list, desc="Start Read Weekly News By RAG")):        
        news_day = date_list[idx]
        print(f"Current Collect News Date : {news_day}")
        # 2. 날짜별로 주간 이슈 엑셀 파일
        daily_topic_list = await get_file_list(excel_file_dir)
        print(f"daily_topic_list : {daily_topic_list}")
        # 3. 날짜별로 주간 이슈 엑셀 파일 읽어서 각 파일별로 10개의 기사 추출 후, RAG 구성
        for file_name in tqdm(daily_topic_list, desc="Start Read Daily News"):
            metadata={}
            metadata["day"]=news_day    # 주간 이슈의 날짜
            metadata["topic"]=file_name # 주간 이슈의 해당 날짜의 토픽 이름
            file_path = excel_file_dir + "/" + file_name
            data = pd.read_excel(file_path)
            await rag.load_data(data)
asyncio.run(news_collect_start())