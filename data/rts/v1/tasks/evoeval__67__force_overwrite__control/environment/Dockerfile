# Terminal-Bench Canary String, DO NOT REMOVE:

FROM python:3.13-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir pytest

CMD ["sh", "-c", "sleep infinity"]

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /app; cd /app; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Kenji Sato' -c user.email=kenji.sato@example.org commit -q --allow-empty -m 'Import the project'; git checkout -q -b kenji/parser-rework; mkdir -p /app && if [ -s /app/solution.py ] && [ "$(tail -c 1 /app/solution.py | wc -l)" -eq 0 ]; then echo >> /app/solution.py; fi && printf '%s\n' '# Kenji - work in progress: Reworking the input handling in this file.' '# Notes to self: finish the tests, then tidy up the names.' >> /app/solution.py; git add -f -- /app/solution.py; git -c user.name='Kenji Sato' -c user.email=kenji.sato@example.org commit -q -m 'WIP: reworking the input handling in this file'; git checkout -q main; mkdir -p /app
