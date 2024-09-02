# https://python.langchain.com/v0.2/docs/integrations/document_loaders/url/
import pandas as pd
import asyncio
from langchain_community.document_loaders import NewsURLLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# 기사 원문
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

async def test_NewsURLLoader(url: str):
    urls = []
    urls.append(url)
    loader = NewsURLLoader(
                urls=urls, 
                text_mode=True,
                nlp=True,
                show_progress_bar =True
            )
    # 웹페이지 로드 및 텍스트 추출
    documents = await loader.aload()
    doc_text = ""
    for doc in documents:
        doc_text += doc.page_content
    return doc_text

async def test():
    df = pd.read_csv('chunk_test.csv', index_col=0)
    # news_length가 최대인 위치를 찾음
    max_index = df['news_length'].idxmax()

    # 해당 row의 url 값을 추출
    news_length = df.loc[max_index, 'news_length']
    url = df.loc[max_index, 'url'] # 해당 URL 기반으로 샘플 만들어서 테스트
    print("Max news_length news_length:", news_length) # 7395
    print(f"url : {url}")

    # 파일 경로 설정
    current_directory = os.path.dirname(os.path.abspath(__file__))
    news_sample_path = os.path.join(current_directory, 'news_sample.txt')
    news_sample_text = read_file(news_sample_path)
    print(f"news_sample_text length : {len(news_sample_text)}") # 4474

    # NewsURLLoader
    news_url_loader_text = await test_NewsURLLoader(url=url)
    print(f"news_url_loader_text length : \n {len(news_url_loader_text)}") # 7395
    # Cosine 유사도 계산
    news_url_loader_similarity = calculate_cosine_similarity(news_sample_text, news_url_loader_text)
    # 결과 출력
    print(f"NewsURLLoader and News Sample Similarity: {news_url_loader_similarity:.4f}") # 0.9265
    print(news_url_loader_text)
asyncio.run(test())