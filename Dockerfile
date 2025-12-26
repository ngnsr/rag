FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

EXPOSE 8501

ENV TOKENIZERS_PARALLELISM=false

CMD ["streamlit", "run", "app/web.py", "--server.port=8501", "--server.address=0.0.0.0"]
