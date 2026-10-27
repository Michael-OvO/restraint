FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    make \
    binutils \
    libasan8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages pytest==8.4.2

WORKDIR /app

COPY src/ /app/src/

COPY suite/test_outputs.py /app/tests/test_outputs.py
