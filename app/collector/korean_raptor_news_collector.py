import os
import pandas as pd
import asyncio
from tqdm import tqdm
from app.raptor.raptor_vectordb_insert import RAPTOR
from app.vectordb.vectordb import get_summary_info

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
async def get_file_list(excel_file_dir: str):
    file_list = []
    absolute_path = os.path.abspath(excel_file_dir)
    if os.path.exists(absolute_path):
        file_list = os.listdir(absolute_path)
    else:
        file_list = ["Directory not found"]
    return file_list

# RAPTOR Collect Process Start
async def news_collect_start():
    raptor = RAPTOR(chunk_size=1000, overlap=200)
    excel_file_dir_list = await get_excel_file_dir_list()
    metadata = {}
    metadata["week"]="8월 5주"

    # 주간 이슈 내의 날짜별 디렉토리를 가져와서 NEWS RAPTOR와 TOPIC RAPTOR 구성
    for idx, excel_file_dir in enumerate(tqdm(excel_file_dir_list, desc="Start Read Weekly News By RAPTOR")):
        news_day = date_list[idx]
        metadata["day"]=news_day

        daily_topic_list = await get_file_list(excel_file_dir)

        # NEWS RAPTOR: 주간 이슈의 특정 일자의 10개의 이슈 중 각 이슈(topic)에 대한 기사들로 RAPTOR를 구성
        for file_idx, file_name in enumerate(tqdm(daily_topic_list, desc="Start Read Daily News")):
            metadata["topic"]=file_name

            file_path = excel_file_dir + "/" + file_name
            data = pd.read_excel(file_path)
            # 각 이슈별로 10개의 뉴스들이 들어감 (하나의 날짜에 총 100개의 기사)
            await raptor.load_data(data=data, metadata=metadata)

        # TOPIC RAPTOR: 일간 뉴스의 각 "topic"에 대한 summarize된 내용(NEWS RAPTOR의 ROOT NODE)을 바탕으로 재귀적 요약 처리
        metadata["topic"]=""
        topic_summary_results = get_summary_info("topic", "news_level")
        topic_leaf_text = []
        for topic_summary in topic_summary_results:
            document = str(topic_summary["document"])
            topic_leaf_text.append(document)
        topic_document = await raptor.insert_additional_info(text_list=topic_leaf_text, metadata=metadata, meta_level="topic_level")
        await raptor.insert_vectordb(text_lists=topic_leaf_text, documents_list=topic_document, metadata=metadata, meta_level="topic_level")
    
    # DAY_RAPTOR: 주간 뉴스의 각 "day"에 대한 summarize된 내용(TOPIC RAPTOR의 ROOT NODE) 을 바탕으로 재귀적 요약 처리
    metadata["day"]=""
    day_summary_results = get_summary_info("day", "topic_level")
    day_leaf_text = []
    for day_summary in day_summary_results:
        document = str(day_summary["document"])
        day_leaf_text.append(document)
    day_document = await raptor.insert_additional_info(text_list=day_leaf_text, metadata=metadata, meta_level="day_level")
    await raptor.insert_vectordb(text_lists=day_leaf_text, documents_list=day_document, metadata=metadata, meta_level="day_level")

asyncio.run(news_collect_start())