FROM skylensage-registry.cn-hangzhou.cr.aliyuncs.com/instances/1124:t-bench-python-3-13

RUN mkdir -p /app/config

COPY app.conf /app/config/app.conf

WORKDIR /app

COPY suite/test_outputs.py /app/tests/test_outputs.py
