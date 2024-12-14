FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app
COPY data/ /app/data/

RUN mkdir -p /app/models

# Make run.sh executable
RUN chmod +x /app/run.sh

# Default command: run the end-to-end script
CMD ["./run.sh"]