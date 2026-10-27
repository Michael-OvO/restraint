FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc-14 \
    make \
    unzip \
    libasan8 \
    libubsan1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/bin/gcc \
    && ln -s /usr/bin/gcc-14 /usr/bin/gcc

RUN pip3 install --break-system-packages pytest==8.4.2

WORKDIR /app

COPY src/ /app/src/

COPY suite/test_outputs.py /app/tests/test_outputs.py
