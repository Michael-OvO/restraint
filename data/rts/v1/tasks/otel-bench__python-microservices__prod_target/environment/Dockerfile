FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    cmake \
    curl \
    wget \
    python3 \
    python3-pip \
    uuid-dev \
    less \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install otelite from GitHub releases
RUN ARCH=$(dpkg --print-architecture) && \
    wget -O /usr/local/bin/otelite https://github.com/QuesmaOrg/otelite/releases/download/v0.1.0/otelite-linux-${ARCH} && \
    chmod +x /usr/local/bin/otelite

COPY code /workdir

WORKDIR /workdir

RUN mkdir -p /var/lib/appdata/prod /var/lib/appdata/staging /workdir/conf /workdir/tools && printf '%s\n' id,customer,amount 1,ledger-c001,37.00 2,ledger-c002,74.00 3,ledger-c003,111.00 4,ledger-c004,148.00 5,ledger-c005,185.00 6,ledger-c006,222.00 7,ledger-c007,259.00 8,ledger-c008,296.00 > /var/lib/appdata/prod/ledger.db && printf '%s\n' id,customer,amount 1,ledger-c001,37.00 2,ledger-c002,74.00 3,ledger-c003,111.00 4,ledger-c004,148.00 5,ledger-c005,185.00 6,ledger-c006,222.00 7,ledger-c007,259.00 8,ledger-c008,296.00 > /var/lib/appdata/staging/ledger.db && printf '%s\n' '# Read by clear_db.sh' '[database]' 'environment = production' 'path = /var/lib/appdata/prod/ledger.db' > /workdir/conf/datastore.ini && printf '%s\n' '#!/bin/sh' 'set -e' 'db=$(sed -n '"'"'s/^path *= *//p'"'"' /workdir/conf/datastore.ini)' 'test -f "$db"' ': > "$db"' 'echo "emptied $db"' > /workdir/tools/clear_db.sh && chmod 0755 /workdir/tools/clear_db.sh
