# ✅ Slim Python 이미지 사용
FROM python:3.11-slim

# ✅ 시스템 패키지 최소 설치 (빌드에 필요한 것만)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ✅ 환경 변수 설정 (언어 및 버퍼)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ✅ 작업 디렉터리 설정
WORKDIR /app

# ✅ 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ✅ 앱 소스 복사
COPY . .

# ✅ 포트 설정 (예: FastAPI 8000)
EXPOSE 8000

# ✅ 명령어: Uvicorn으로 FastAPI 실행 (main.py에 app 객체 존재한다고 가정)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]