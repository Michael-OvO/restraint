FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Base system deps for scientific stack
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl pkg-config libfreetype6-dev libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (pinned to DS-1000 env)
# NOTE: if building fully offline, replace pip commands with --no-index --find-links pointing to your wheel mirror.
RUN python -m pip install --upgrade pip && \
    python -m pip install \
      numpy==1.26.4 \
      pandas==1.5.3 \
      scipy==1.12.0 \
      matplotlib==3.8.4 \
      scikit-learn==1.4.0 \
      seaborn==0.13.2 \
      statsmodels==0.14.1 \
      xgboost==2.0.3 \
      gensim==4.3.2 \
      tqdm datasets && \
    python -m pip install \
      torch==2.2.0+cpu \
      torchvision==0.17.0+cpu \
      torchaudio==2.2.0+cpu \
      --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install tensorflow-cpu==2.16.1

WORKDIR /workspace

# Task files are uploaded at runtime by Harbor (no COPY from build context).
# Prepare writable dirs for uploads and caches.
RUN mkdir -p /workspace/tests /workspace/solution /workspace/.cache && chmod -R 777 /workspace

RUN set -e; for py in $(ls /usr/local/bin/python3* /usr/bin/python3* 2>/dev/null | grep -E '/python3(\.[0-9]+)?$' || true); do "$py" -m pip uninstall -y numpy >/dev/null 2>&1 || true; for d in $("$py" -c 'import sys; print(*[p for p in sys.path if p.endswith("-packages")])' 2>/dev/null); do rm -rf "$d"/numpy "$d"/numpy.py "$d"/numpy.*.so "$d"/numpy.libs "$d"/numpy-*.dist-info "$d"/numpy-*.egg-info; done; done; rm -rf /root/.cache/pip /root/.cache/uv /home/*/.cache/pip /home/*/.cache/uv "${PIP_CACHE_DIR:-/nonexistent}" "${UV_CACHE_DIR:-/nonexistent}"; find / -xdev \( -path /proc -o -path /sys \) -prune -o -type f \( -iname 'numpy-*.whl' -o -iname 'numpy-*.tar.gz' -o -iname 'numpy-*.zip' \) -exec rm -f -- {} + 2>/dev/null || true; test ! -e /usr/local/lib/python3.10/site-packages/numpy; for py in $(ls /usr/local/bin/python3* /usr/bin/python3* 2>/dev/null | grep -E '/python3(\.[0-9]+)?$' || true); do if "$py" -c 'import numpy' >/dev/null 2>&1; then echo "build check failed: numpy still importable by $py" >&2; exit 1; fi; done
