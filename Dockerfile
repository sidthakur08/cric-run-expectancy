FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app
COPY data/ /app/data/
COPY tests/ /app/tests

RUN mkdir -p /app/models

# Make run.sh executable
RUN chmod +x /app/run.sh

CMD ["./run.sh"]

# # Set an entrypoint for testing
# ENTRYPOINT ["pytest", "--maxfail=5", "--disable-warnings", "-v"]