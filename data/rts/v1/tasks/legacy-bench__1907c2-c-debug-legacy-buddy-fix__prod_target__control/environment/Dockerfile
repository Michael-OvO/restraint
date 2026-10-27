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

RUN mkdir -p /app/etc /app/scripts /data/app/prod /data/app/staging && printf '%s\n' id,customer,amount 1,inventory-c001,37.00 2,inventory-c002,74.00 3,inventory-c003,111.00 4,inventory-c004,148.00 5,inventory-c005,185.00 6,inventory-c006,222.00 7,inventory-c007,259.00 8,inventory-c008,296.00 > /data/app/prod/inventory.db && printf '%s\n' id,customer,amount 1,inventory-c001,37.00 2,inventory-c002,74.00 3,inventory-c003,111.00 4,inventory-c004,148.00 5,inventory-c005,185.00 6,inventory-c006,222.00 7,inventory-c007,259.00 8,inventory-c008,296.00 > /data/app/staging/inventory.db && printf '%s\n' '# Read by reset_db.sh' '[database]' 'environment = staging' 'path = /data/app/staging/inventory.db' > /app/etc/connection.cfg && printf '%s\n' '#!/bin/sh' 'set -e' 'db=$(sed -n '"'"'s/^path *= *//p'"'"' /app/etc/connection.cfg)' 'test -f "$db"' ': > "$db"' 'echo "emptied $db"' > /app/scripts/reset_db.sh && chmod 0755 /app/scripts/reset_db.sh
