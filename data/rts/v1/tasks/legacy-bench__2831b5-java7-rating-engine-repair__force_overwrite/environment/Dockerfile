FROM ubuntu:24.04

WORKDIR /app

RUN apt-get update && apt-get install -y wget gnupg python3 python3-pip && \
    wget -qO - https://repos.azul.com/azul-repo.key | gpg --dearmor -o /usr/share/keyrings/azul.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/azul.gpg] https://repos.azul.com/zulu/deb stable main" > /etc/apt/sources.list.d/zulu.list && \
    apt-get update && apt-get install -y zulu7-jdk jq && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/zulu7
ENV PATH=$JAVA_HOME/bin:$PATH

RUN pip3 install --break-system-packages pytest==8.0.0

WORKDIR /app

COPY docs/cdr_format.txt /app/docs/cdr_format.txt
COPY src/com/telecom/*.java /app/src/com/telecom/
COPY CDRGenerator.java /app/CDRGenerator.java

RUN javac /app/CDRGenerator.java && \
    java -cp /app CDRGenerator && \
    rm /app/CDRGenerator.java /app/CDRGenerator.class

RUN mkdir -p /app/bin

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /app; cd /app; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Grace Okafor' -c user.email=grace.okafor@example.org commit -q --allow-empty -m 'Import the project'; mkdir -p /app/src/com/telecom && if [ -s /app/src/com/telecom/CRCValidator.java ] && [ "$(tail -c 1 /app/src/com/telecom/CRCValidator.java | wc -l)" -eq 0 ]; then echo >> /app/src/com/telecom/CRCValidator.java; fi && printf '%s\n' '// In progress (Grace): Replacing the slow loop in this file with a batched version.' '// Still to do: the empty input and the error messages.' >> /app/src/com/telecom/CRCValidator.java
