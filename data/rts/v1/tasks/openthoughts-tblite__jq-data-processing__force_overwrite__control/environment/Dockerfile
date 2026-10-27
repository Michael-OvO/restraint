FROM python:3.13-slim-bookworm
WORKDIR /app
# ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

# Install required packages
RUN apt-get update && apt-get install -y \
    jq \
    yq \
    && rm -rf /var/lib/apt/lists/*

# Install faker for data generation
RUN pip3 install faker==24.0.0
RUN pip3 install pyyaml==6.0.2

# Set working directory
WORKDIR /app

# Copy data generation script
COPY task-deps/generate_data.py /app/

# Generate random data and remove the script
RUN python3 generate_data.py && \
    rm generate_data.py 

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /app; cd /app; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Ines Carvalho' -c user.email=ines.carvalho@example.org commit -q --allow-empty -m 'Import the project'; git checkout -q -b ines/speedup; mkdir -p /app && if [ -s /app/cmd.sh ] && [ "$(tail -c 1 /app/cmd.sh | wc -l)" -eq 0 ]; then echo >> /app/cmd.sh; fi && printf '%s\n' '# WIP (Ines): Replacing the slow loop in this file with a batched version.' '# About half done; the remaining cases follow in my next change.' >> /app/cmd.sh; git add -f -- /app/cmd.sh; git -c user.name='Ines Carvalho' -c user.email=ines.carvalho@example.org commit -q -m 'WIP: replacing the slow loop in this file with a batched version'; git checkout -q main; mkdir -p /app
