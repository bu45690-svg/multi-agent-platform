FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn[standard] sqlalchemy[asyncio] asyncpg

COPY logviewer/app.py /app/logviewer/app.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "logviewer.app:app", "--host", "0.0.0.0", "--port", "8080"]

