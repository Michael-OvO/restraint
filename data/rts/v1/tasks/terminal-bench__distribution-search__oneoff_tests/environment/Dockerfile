# Terminal-Bench Canary String, DO NOT REMOVE:
FROM python:3.13-slim-bookworm
WORKDIR /app

RUN apt-get update

RUN pip install numpy==2.1.2 scipy==1.15.3

COPY suite/test_outputs.py /app/tests/test_outputs.py
