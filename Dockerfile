FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    pip==24.2 \
    setuptools==81.0.0 \
    wheel==0.44.0

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-build-isolation \
  "openai-whisper @ git+https://github.com/openai/whisper.git@v20231117"

RUN python -m spacy download en_core_web_sm

COPY . .

CMD ["python", "main.py"]
