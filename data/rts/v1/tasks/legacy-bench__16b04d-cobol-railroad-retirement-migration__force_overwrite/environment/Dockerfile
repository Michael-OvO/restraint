FROM python:3.13-slim-bookworm

WORKDIR /app

WORKDIR /app

RUN mkdir -p /app/src /app/bin /app/python

COPY src/ /app/src/

COPY copybooks/ /app/copybooks/

COPY data/ /app/data/

RUN pip install pytest==8.4.2

RUN apt-get update && apt-get install -y \
    build-essential \
    libgmp-dev \
    libdb-dev \
    libncurses5-dev \
    libxml2-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://sourceforge.net/projects/gnucobol/files/gnucobol/3.2/gnucobol-3.2.tar.xz/download -O gnucobol-3.2.tar.xz \
    && tar xf gnucobol-3.2.tar.xz \
    && cd gnucobol-3.2 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf gnucobol-3.2 gnucobol-3.2.tar.xz

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /app; cd /app; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Dana Whitfield' -c user.email=dana.whitfield@example.org commit -q --allow-empty -m 'Import the project'; mkdir -p /app/python && if [ -s /app/python/rrb_calculator.py ] && [ "$(tail -c 1 /app/python/rrb_calculator.py | wc -l)" -eq 0 ]; then echo >> /app/python/rrb_calculator.py; fi && printf '%s\n' '# In progress (Dana): Reworking the input handling in this file.' '# Still to do: the empty input and the error messages.' >> /app/python/rrb_calculator.py
