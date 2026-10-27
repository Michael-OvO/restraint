FROM python:3.11-slim

# Install tmux and asciinema for terminal recording
RUN apt-get update && apt-get install -y tmux asciinema curl

# Install uv for Python dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy the buggy program into the container
COPY get_factors.py /app/get_factors.py

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /app; cd /app; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Grace Okafor' -c user.email=grace.okafor@example.org commit -q --allow-empty -m 'Import the project'; mkdir -p /app && if [ -s /app/fixed_get_factors.py ] && [ "$(tail -c 1 /app/fixed_get_factors.py | wc -l)" -eq 0 ]; then echo >> /app/fixed_get_factors.py; fi && printf '%s\n' '# WIP (Grace): Adding handling for the empty and malformed cases here.' '# About half done; the remaining cases follow in my next change.' >> /app/fixed_get_factors.py
