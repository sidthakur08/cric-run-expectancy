# Base image with Python 3.10
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY app/ /app
COPY data/ /app/data/
COPY tests/ /app/tests

# Create a models folder
RUN mkdir -p /app/models

# Make run.sh executable
RUN chmod +x /app/run.sh

# Set the default command to run the pipeline
CMD ["bash", "run.sh"]