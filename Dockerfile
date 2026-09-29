# Public ECR mirror avoids Docker Hub rate limits in hackathon accounts
FROM public.ecr.aws/docker/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 RUNTIME_DIR=/srv/runtime
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://pypi.org/simple -r requirements.txt

COPY app app
COPY web web
COPY starter starter
COPY data data
COPY scripts scripts
COPY recommender recommender

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
# One worker: SQLite state + per-customer locks live in this process (threadpool handles concurrency)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 75"]
