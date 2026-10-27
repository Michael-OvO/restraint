FROM skylensage-registry.cn-hangzhou.cr.aliyuncs.com/instances/1124:t-bench-ubuntu-24-04

RUN mkdir -p /app/config

COPY sample.conf /app/config/

WORKDIR /app

COPY suite/test_outputs.py /app/tests/test_outputs.py
