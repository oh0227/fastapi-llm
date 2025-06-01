# ✅ 최소 base 이미지 사용
FROM python:3.11-slim

# ✅ 시스템 패키지 최소 설치
RUN apt-get update && apt-get install -y \
    gcc build-essential libpq-dev curl git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# ✅ 의존성 최소화된 requirements.txt 사용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
WORKDIR /

CMD ["python", "main.py"]