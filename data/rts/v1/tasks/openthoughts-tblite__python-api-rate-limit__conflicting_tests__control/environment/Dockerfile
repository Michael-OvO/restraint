FROM python:3.13-slim-bookworm

WORKDIR /app

# Basic OS dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY src/ /app/src

# Default command (Harbor overrides this, but Docker requires one)
CMD ["bash"]

COPY suite/test_outputs.py /app/tests/test_outputs.py
