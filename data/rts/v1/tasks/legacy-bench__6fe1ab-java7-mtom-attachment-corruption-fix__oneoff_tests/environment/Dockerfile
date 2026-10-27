FROM python:3.13-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y wget gnupg junit4 && \
    wget -qO - https://repos.azul.com/azul-repo.key | gpg --dearmor -o /usr/share/keyrings/azul.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/azul.gpg] https://repos.azul.com/zulu/deb stable main" > /etc/apt/sources.list.d/zulu.list && \
    apt-get update && apt-get install -y zulu7-jdk && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/zulu7
ENV PATH=$JAVA_HOME/bin:$PATH

RUN pip install pytest==8.4.2

WORKDIR /app

COPY src/ /app/src/
COPY data/ /app/data/

RUN mkdir -p /app/bin /app/output

COPY suite/test_outputs.py /app/tests/test_outputs.py
