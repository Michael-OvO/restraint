FROM python:3.13-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential libgmp-dev libdb-dev libncurses5-dev libxml2-dev wget \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget https://ftp.gnu.org/gnu/gnucobol/gnucobol-3.2.tar.xz && \
    tar xf gnucobol-3.2.tar.xz && \
    cd gnucobol-3.2 && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/gnucobol-3.2 /tmp/gnucobol-3.2.tar.xz

RUN pip install pytest==8.0.0

RUN mkdir -p /app/copybooks /app/data /app/output

COPY copybooks/*.cpy /app/copybooks/
COPY src/*.cob /app/
COPY generate_data.py /tmp/generate_data.py
RUN python3 /tmp/generate_data.py && rm /tmp/generate_data.py

WORKDIR /app

COPY suite/test_outputs.py /app/tests/test_outputs.py
