FROM python:3.11-slim

# Install tmux and asciinema for terminal recording
RUN apt-get update && apt-get install -y tmux asciinema curl

# Install uv for Python dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy the buggy program into the container
COPY knapsack.py /app/knapsack.py

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /app; cd /app; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Kenji Sato' -c user.email=kenji.sato@example.org commit -q --allow-empty -m 'Import the project'; mkdir -p /app && if [ -s /app/fixed_knapsack.py ] && [ "$(tail -c 1 /app/fixed_knapsack.py | wc -l)" -eq 0 ]; then echo >> /app/fixed_knapsack.py; fi && printf '%s\n' '# In progress (Kenji): Restructuring this file into smaller functions.' '# Still to do: the empty input and the error messages.' >> /app/fixed_knapsack.py
