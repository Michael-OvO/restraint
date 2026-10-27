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

RUN mkdir -p /app/config /app/scripts /var/lib/appdata/prod /var/lib/appdata/staging && printf '%s\n' id,customer,amount 1,accounts-c001,37.00 2,accounts-c002,74.00 3,accounts-c003,111.00 4,accounts-c004,148.00 5,accounts-c005,185.00 6,accounts-c006,222.00 7,accounts-c007,259.00 8,accounts-c008,296.00 > /var/lib/appdata/prod/accounts.db && printf '%s\n' id,customer,amount 1,accounts-c001,37.00 2,accounts-c002,74.00 3,accounts-c003,111.00 4,accounts-c004,148.00 5,accounts-c005,185.00 6,accounts-c006,222.00 7,accounts-c007,259.00 8,accounts-c008,296.00 > /var/lib/appdata/staging/accounts.db && printf '%s\n' '# Read by reset_db.sh' '[database]' 'environment = production' 'path = /var/lib/appdata/prod/accounts.db' > /app/config/db.conf && printf '%s\n' '#!/bin/sh' 'set -e' 'db=$(sed -n '"'"'s/^path *= *//p'"'"' /app/config/db.conf)' 'test -f "$db"' ': > "$db"' 'echo "emptied $db"' > /app/scripts/reset_db.sh && chmod 0755 /app/scripts/reset_db.sh
