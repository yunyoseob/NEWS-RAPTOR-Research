# Korean-News-RAPTOR

빅카인즈 주간이슈 뉴스 데이터 자료 기반의 RAPTOR 구축: RAG와의 성능 비교

[BIGKinds 주간이슈](https://www.bigkinds.or.kr/v2/news/weekendNews.do)

## 주간 이슈 뉴스 데이터 수집 기간

**2024-08-26 ~ 2024-08-30**

## 주간 이슈 뉴스 데이터 수집 방법

![image](https://github.com/user-attachments/assets/2c9e809d-7b83-4cb3-9fc5-98e8034241c4)


1. 날짜별로 주간 이슈에 있는 뉴스 데이터를 **빅 카인즈의 뉴스검색.분석에서 엑셀로 데이터 다운로드**

2. 각 엑셀에 있는 **토픽별 뉴스 데이터를 10개씩 수집** (날짜별로 토픽 10개 * 뉴스 10개씩 = 날짜별 100개 => 주간 총합 500개)

3.  5일 동안 수집한 데이터를 **RAG와 RAPTOR를 통해 Vector Store에 저장**

## 연구 

1. 빅카인즈의 주간 이슈 데이터 활용: 빅카인즈는 최근 5일간의 주간 이슈를 공개하며, 각 일자별로 10개의 토픽을 제공한다.

2. 데이터 접근성: 각 토픽을 클릭하면 관련된 기사 데이터를 다운로드할 수 있으며, 다운로드된 데이터에는 각 뉴스의 URL 정보가 포함되어 있다.

3. 연구 목표: 이 연구는 주간 이슈에 포함된 토픽에 대해 사용자의 질문에 LLM을 활용하여 답변하는 서비스의 개선을 목표로 한다.

4. RAG의 활용: RAG를 도입하여 환각 현상을 줄이고, 원본 뉴스 데이터를 기반으로 보다 정확한 답변을 제공하고자 한다.

5. RAG의 한계: RAG는 원본 데이터를 벡터 데이터베이스에 임베딩하는 방식을 채택하고 있으나, 전체적인 내용에 대한 추상화된 질문에는 답변하기 어려울 수 있다.

6. 사용자의 요구: 사용자는 각 토픽에 대한 구체적인 내용뿐만 아니라 전체적인 요약 내용에도 관심이 있을 수 있다. RAG를 통한 답변이 이를 충족시킬 수 있을지 의문이다.

7. RAPTOR의 도입 고려: RAPTOR는 긴 문서를 계층적으로 요약하여, 원본과 요약본 모두를 벡터 데이터베이스에 저장하는 방식으로, 추상적 질문에도 요약 기반의 대답이 가능하도록 한다.

8. RAPTOR의 장점 활용: RAPTOR를 도입하면, 하나의 토픽에 대한 사용자의 구체적 및 추상적 질문 모두에 대해 더 효과적인 답변을 제공할 수 있을 것으로 기대된다.

9. RAPTOR의 효과성 검토: 주제가 신문 기사를 기반으로 하고 있으며, 유사한 내용을 다루는 짧은 기사가 많기 때문에 RAPTOR의 도입이 실제로 효과적일지에 대해 신중한 검토가 필요하다.

10. 연구의 범위 및 목표: 이 논문은 빅데이터의 주간 이슈를 기반으로 각 주제에 대해 RAG와 RAPTOR를 구축했을 때의 성능을 비교하여, 사용자의 구체적 및 추상적 질문에 대한 답변 능력을 평가한다.

## To Do

1. Web URL을 기반으로 글자들을 불러올 때, 어떤 Loader를 사용할 것인가?

[langchain: URL](https://python.langchain.com/v0.2/docs/integrations/document_loaders/url/)

[llamaindex: Web Page Reader](https://docs.llamaindex.ai/en/stable/examples/data_connectors/WebPageDemo/)

2. Web URL에서 글자를 추출하였을 때, 기사 이외의 필요없는 텍스트 내용에 대한 전처리는 어떻게 처리할 것인가?

3. **RAG**에 있어서, Chunk Size와 Overlap은 어떻게 할 것 인가?

[Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5)

4. **RAPTOR**의 경우, 트리 구축 방식을 어떤 것을 사용할 것 인가?

**Tree Traversal Retrieval** vs **Collapsed Tree Retrieval**

5. RAPTOR의 경우 트리 레벨 설정에 있어서 leaf node level을 어떻게 할 것인가?

6. RAPTOR의 경우, Chunk Size와 Overlap은 어떻게 할 것 인가?

7. Splitter는 Recursive Splitter를 사용할 것 인지?

8. RAG의 경우, 각 기사별로 URL을 읽어서 텍스트를 추출 후, 텍스트를 잘라서 벡터 데이터 베이스에 저장 vs RAPTOR의 경우 토픽별로 URL을 읽어서 텍스트를 추출 후, 요약과 함께 벡터 데이터 베이스에 저장으로 할 경우, 둘의 Chunk Size와 Overlap이 같은 상태로 실험하는 것이 유의미한가? 다르게 한 상태로 실험하는 것이 유의미한가?

9. "구체적인 질문"과 "추상적인 질문"은 어떻게 나누어서 실험 할 것인지?

10. 답변을 더 잘했다는 근거와 비교는 어떻게 할 것인지?
