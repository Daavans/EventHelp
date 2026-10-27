FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Phase 1: run the smoke test. Will become the Streamlit app in a later phase.
CMD ["python", "hello.py"]
