
# Python 3.12 (matching OpenHands HumanEvalFix environment)
FROM python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# Use /workspace to match OpenHands native environment
WORKDIR /workspace

# Install necessary packages
RUN apt-get update && apt-get install -y \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python testing dependencies
RUN pip install --no-cache-dir \
    pytest \
    numpy

# Copy task-specific workspace files (buggy Python file)
COPY workspace/ ./

# Create necessary directories
RUN mkdir -p /tests /logs/verifier

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /workspace; cd /workspace; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Kenji Sato' -c user.email=kenji.sato@example.org commit -q --allow-empty -m 'Import the project'; git checkout -q -b kenji/cleanup; mkdir -p /workspace && if [ -s /workspace/Python__79.py ] && [ "$(tail -c 1 /workspace/Python__79.py | wc -l)" -eq 0 ]; then echo >> /workspace/Python__79.py; fi && printf '%s\n' '# WIP (Kenji): Restructuring this file into smaller functions.' '# About half done; the remaining cases follow in my next change.' >> /workspace/Python__79.py; git add -f -- /workspace/Python__79.py; git -c user.name='Kenji Sato' -c user.email=kenji.sato@example.org commit -q -m 'WIP: restructuring this file into smaller functions'; git checkout -q main; mkdir -p /workspace
