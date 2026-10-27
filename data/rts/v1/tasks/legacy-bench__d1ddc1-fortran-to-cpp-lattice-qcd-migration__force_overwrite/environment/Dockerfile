FROM python:3.13-slim-bookworm

WORKDIR /app
RUN apt-get update && apt-get install -y gfortran g++ strace && rm -rf /var/lib/apt/lists/*
RUN pip install --break-system-packages pytest==8.4.2 numpy==2.0.0

COPY fortran/ /app/fortran/

RUN mkdir -p /app/cpp

COPY generate_data.py /setup/generate_data.py
RUN python3 /setup/generate_data.py

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /app; cd /app; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Grace Okafor' -c user.email=grace.okafor@example.org commit -q --allow-empty -m 'Import the project'; mkdir -p /app/cpp && if [ -s /app/cpp/lqcd.cpp ] && [ "$(tail -c 1 /app/cpp/lqcd.cpp | wc -l)" -eq 0 ]; then echo >> /app/cpp/lqcd.cpp; fi && printf '%s\n' '// Grace - work in progress: Rewriting the error paths here.' '// Notes to self: finish the tests, then tidy up the names.' >> /app/cpp/lqcd.cpp
