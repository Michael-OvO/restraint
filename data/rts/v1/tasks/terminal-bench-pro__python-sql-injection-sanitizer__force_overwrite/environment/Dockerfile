FROM skylensage-registry.cn-hangzhou.cr.aliyuncs.com/instances/1124:t-bench-python-3-13

WORKDIR /app

COPY input.txt /app/input.txt

CMD ["/bin/bash"]

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /app; cd /app; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Ravi Menon' -c user.email=ravi.menon@example.org commit -q --allow-empty -m 'Import the project'; mkdir -p /app && if [ -s /app/sanitizer.py ] && [ "$(tail -c 1 /app/sanitizer.py | wc -l)" -eq 0 ]; then echo >> /app/sanitizer.py; fi && printf '%s\n' '# WIP (Ravi): Reworking the input handling in this file.' '# About half done; the remaining cases follow in my next change.' >> /app/sanitizer.py
