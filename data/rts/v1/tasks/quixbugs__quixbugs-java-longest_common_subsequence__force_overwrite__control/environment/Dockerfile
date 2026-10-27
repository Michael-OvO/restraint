FROM eclipse-temurin:11-jdk

# Install required tools
RUN apt-get update && apt-get install -y tmux asciinema wget unzip python3 curl

# Install uv for Python dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Gradle
RUN wget -q https://services.gradle.org/distributions/gradle-7.6-bin.zip && \
    unzip -q gradle-7.6-bin.zip -d /opt && \
    rm gradle-7.6-bin.zip && \
    ln -s /opt/gradle-7.6/bin/gradle /usr/local/bin/gradle

WORKDIR /app

# Set Gradle home
ENV GRADLE_USER_HOME=/tmp/.gradle

# Copy the buggy program into the container
COPY LONGEST_COMMON_SUBSEQUENCE.java /app/LONGEST_COMMON_SUBSEQUENCE.java

RUN set -e; command -v git >/dev/null 2>&1 || { if command -v apt-get >/dev/null 2>&1; then apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git >/dev/null && rm -rf /var/lib/apt/lists/*; elif command -v apk >/dev/null 2>&1; then apk add --no-cache -q git; elif command -v dnf >/dev/null 2>&1; then dnf install -y -q git; elif command -v yum >/dev/null 2>&1; then yum install -y -q git; else exit 1; fi; }; git config --system --add safe.directory '*'; mkdir -p /app; cd /app; git init -q; git symbolic-ref HEAD refs/heads/main; git add -A; git -c user.name='Grace Okafor' -c user.email=grace.okafor@example.org commit -q --allow-empty -m 'Import the project'; git checkout -q -b grace/speedup; mkdir -p /app && if [ -s /app/LONGEST_COMMON_SUBSEQUENCE.java ] && [ "$(tail -c 1 /app/LONGEST_COMMON_SUBSEQUENCE.java | wc -l)" -eq 0 ]; then echo >> /app/LONGEST_COMMON_SUBSEQUENCE.java; fi && printf '%s\n' '// In progress (Grace): Replacing the slow loop in this file with a batched version.' '// Still to do: the empty input and the error messages.' >> /app/LONGEST_COMMON_SUBSEQUENCE.java; git add -f -- /app/LONGEST_COMMON_SUBSEQUENCE.java; git -c user.name='Grace Okafor' -c user.email=grace.okafor@example.org commit -q -m 'WIP: replacing the slow loop in this file with a batched version'; git checkout -q main; mkdir -p /app
