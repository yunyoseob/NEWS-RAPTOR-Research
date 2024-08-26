import os
import pandas as pd
import asyncio
from tqdm import tqdm
from app.rag.rag_vectordb_insert import RAG
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT_DIR = os.getenv('PROJECT_ROOT_DIR')
date_list = ["20240715","20240716","20240717","20240718","20240719"]

async def get_excel_file_dir_list():
    data_dir = os.path.join(PROJECT_ROOT_DIR, 'data')
    excel_file_dir_list=[]
    for date in date_list:
        excel_file_dir = data_dir + "/" + date
        excel_file_dir_list.append(excel_file_dir)
    return excel_file_dir_list

async def get_file_list(excel_file_dir: str):
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
        news_date = date_list[idx]
        print(f"Current Collect News Date : {news_date}")
        # 2. 날짜별로 주간 이슈 엑셀 파일
        daily_topic_list = await get_file_list(excel_file_dir)
        print(f"daily_topic_list : {daily_topic_list}")
        # 3. 날짜별로 주간 이슈 엑셀 파일 읽어서 각 파일별로 10개의 기사 추출 후, RAG 구성
        for file_name in tqdm(daily_topic_list, desc="Start Read Daily News"):
            file_path = excel_file_dir + "/" + file_name
            data = pd.read_excel(file_path)
            await rag.load_data(data)
asyncio.run(news_collect_start())