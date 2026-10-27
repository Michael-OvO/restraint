FROM python:3.13-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY app/ /app/

ENV PYTHONPATH=/app

COPY suite/test_outputs.py /app/tests/test_outputs.py
