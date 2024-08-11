#!/bin/bash

# .env 파일 로드
export $(grep -v '^#' .env | xargs)

# PROJECT_ROOT_DIR이 설정되어 있는지 확인
if [ -z "$PROJECT_ROOT_DIR" ]; then
  echo "PROJECT_ROOT_DIR 환경 변수가 설정되지 않았습니다."
  exit 1
fi

# 프로젝트 루트 디렉토리로 이동
cd "$PROJECT_ROOT_DIR" || exit

# Python 스크립트 실행
python3 "$PROJECT_ROOT_DIR/app/collector/korean_news_collector.py"
