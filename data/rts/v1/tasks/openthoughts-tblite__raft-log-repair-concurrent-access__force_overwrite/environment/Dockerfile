# Hint from Snorkel
# Base Dockerfile or image that fully sets up the environment, including all required tools, resources, and dependencies.
# It should be reproducible, lightweight, and run without privileged mode.

FROM python:3.13-slim-bookworm

WORKDIR /app

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /app; cd /app; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Dana Whitfield' -c user.email=dana.whitfield@example.org commit -q --allow-empty -m 'Import the project'; mkdir -p /app && if [ -s /app/solution.py ] && [ "$(tail -c 1 /app/solution.py | wc -l)" -eq 0 ]; then echo >> /app/solution.py; fi && printf '%s\n' '# In progress (Dana): Replacing the slow loop in this file with a batched version.' '# Still to do: the empty input and the error messages.' >> /app/solution.py
