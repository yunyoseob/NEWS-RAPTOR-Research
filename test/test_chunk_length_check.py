import os
from app.config import get_settings
from langchain_community.document_loaders import NewsURLLoader
import asyncio
import pandas as pd

class SaveChunkTest:
    def __init__(self):
        self.config = get_settings()
        self.date_list = ["20240826","20240827","20240828","20240829","20240830"]
        self.df = pd.DataFrame(columns=['day', 'topic', 'news_count', 'url', 'news_length'])
        self.index = 0

    async def get_excel_file_dir_list(self):
        data_dir = os.path.join(self.config.PROJECT_ROOT_DIR, 'data')
        excel_file_dir_list=[]
        for date in self.date_list:
            excel_file_dir = data_dir + "/" + date
            excel_file_dir_list.append(excel_file_dir)
        return excel_file_dir_list

    async def get_file_list(self, excel_file_dir: str) -> list[str]:
        file_list = []
        absolute_path = os.path.abspath(excel_file_dir)
        if os.path.exists(absolute_path):
            file_list = os.listdir(absolute_path)
        else:
            file_list = ["Directory not found"]
        return file_list
    
    async def get_documents_length_by_news_url(self, url):
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
            if documents:
                # 첫 번째 문서의 텍스트 가져오기
                document_text = documents[0].page_content
                # 글자 수 계산
                news_length = len(document_text)
            else:
                news_length = 0
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            news_length = 0
        return news_length

    async def load_data(self, day, topic, news_count, data):
        news_data_url = data['URL']

        news_cnt = 0
        for idx, url  in enumerate(news_data_url):
            if news_cnt >= 10:
                break
            if pd.notna(url):  # Check if url is not NaN
                try:
                    # Document Loader
                    news_length = await self.get_documents_length_by_news_url(url)
                    print(f"Save Chunk Info : day : {day}, topic : {topic}, news_count : {news_count}, url : {url}, news_length : {news_length}")
                    if news_length > 0:
                        self.df.loc[self.index] = [day, topic, news_count, url, news_length]
                        self.index = self.index + 1
                        news_cnt += 1
                except Exception as e:
                    print(f"Failed to process URL {url}: {e}")
                    pass

    async def chunk_list_check(self):
        excel_file_dir_list = await self.get_excel_file_dir_list()
        # 1. 날짜 디렉토리
        for idx, excel_file_dir in enumerate(excel_file_dir_list):        
            news_day = self.date_list[idx]
            print(f"Current Collect News Date : {news_day}")
            # 2. 날짜별로 주간 이슈 엑셀 파일
            daily_topic_list = await self.get_file_list(excel_file_dir)
            # 3. 날짜별로 주간 이슈 엑셀 파일 읽어서 각 파일별로 10개의 기사 추출 후, RAG 구성
            for file_name in (daily_topic_list):
                file_path = excel_file_dir + "/" + file_name
                data = pd.read_excel(file_path)
                # 4. 각 토픽 엑셀을 읽어서 날짜, 토픽, 뉴스의 기사 개수, 토픽의 뉴스 기사들의 평균 텍스트 길이를 저장 후 출력
                news_count = len(data)
                await self.load_data(day=news_day, topic=file_name, news_count=news_count, data=data)
        self.df.to_csv('chunk_test.csv', index=True)
        return 1

class GetChunkTest:
    def __init__(self):
        self.df = pd.read_csv('chunk_test.csv', index_col=0)
    def save_result(self, column: str):
        news_count_stats = {
            'mean': self.df[column].mean(),
            'max': self.df[column].max(),
            'min': self.df[column].min(),
            'std': self.df[column].std(),
        }
        stats_text = (
            f"{column} Statistics:\n"
            f"Mean: {news_count_stats['mean']}\n"
            f"Max: {news_count_stats['max']}\n"
            f"Min: {news_count_stats['min']}\n"
            f"Standard Deviation: {news_count_stats['std']}\n"
        )

        # 저장할 경로 설정
        output_dir = '../result'
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{column}.log"
        output_file_path = os.path.join(output_dir, file_name)

        # 파일로 저장
        with open(output_file_path, 'w') as file:
            file.write(stats_text)

        print(f"Statistics saved to {output_file_path}")
        return 1

async def test():
    chunkTest = SaveChunkTest()
    save_result =  await chunkTest.chunk_list_check()
    if save_result == 1:
        getChunkTest = GetChunkTest()
        getChunkTest.save_result(column='news_count')
        getChunkTest.save_result(column='news_length')

asyncio.run(test())