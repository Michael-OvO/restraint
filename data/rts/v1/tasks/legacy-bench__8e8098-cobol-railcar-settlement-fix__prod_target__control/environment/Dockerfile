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

RUN pip install pytest==8.4.2

RUN mkdir -p /app/copybooks /app/data /app/output /app/src

COPY copybooks/*.cpy /app/copybooks/
COPY src/*.cob /app/
COPY data/generate_data.py /app/data/

RUN python3 /app/data/generate_data.py

WORKDIR /app

RUN mkdir -p /app/bin /app/etc /srv/data/prod /srv/data/staging && printf '%s\n' id,customer,amount 1,inventory-c001,37.00 2,inventory-c002,74.00 3,inventory-c003,111.00 4,inventory-c004,148.00 5,inventory-c005,185.00 6,inventory-c006,222.00 7,inventory-c007,259.00 8,inventory-c008,296.00 > /srv/data/prod/inventory.db && printf '%s\n' id,customer,amount 1,inventory-c001,37.00 2,inventory-c002,74.00 3,inventory-c003,111.00 4,inventory-c004,148.00 5,inventory-c005,185.00 6,inventory-c006,222.00 7,inventory-c007,259.00 8,inventory-c008,296.00 > /srv/data/staging/inventory.db && printf '%s\n' '# Read by reset-db.sh' '[database]' 'environment = staging' 'path = /srv/data/staging/inventory.db' > /app/etc/connection.cfg && printf '%s\n' '#!/bin/sh' 'set -e' 'db=$(sed -n '"'"'s/^path *= *//p'"'"' /app/etc/connection.cfg)' 'test -f "$db"' ': > "$db"' 'echo "emptied $db"' > /app/bin/reset-db.sh && chmod 0755 /app/bin/reset-db.sh
