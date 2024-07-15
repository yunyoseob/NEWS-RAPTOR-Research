# 베이스 이미지로 Python 3.10 사용
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 설치
RUN pip install poetry
RUN poetry config virtualenvs.create true

# Poetry 설정 파일과 프로젝트 파일을 컨테이너에 복사
COPY pyproject.toml poetry.lock /app/

# 프로젝트 종속성 설치
RUN poetry install --no-root --no-interaction --no-ansi

# 소스 코드 복사
COPY app /app

# Milvus 서버 시작 스크립트 설정
COPY .entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 컨테이너 시작 시 실행할 명령 설정
ENTRYPOINT ["/entrypoint.sh"]