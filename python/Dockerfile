FROM python:3.12-slim

LABEL org.opencontainers.image.title="AgentKnowledgeOS"
LABEL org.opencontainers.image.authors="zhicong992-droid"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.created="2026-05-19"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-chi-sim poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
