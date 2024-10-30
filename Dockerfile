FROM python:3.11-slim AS project_env

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
    build-essential \
    gcc \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
    && pip install -r requirements.txt

# Create a separate stage to reduce image size
FROM python:3.11-slim AS runtime_env

# Copy dependencies from the previous stage
COPY --from=project_env /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=project_env /usr/local/bin /usr/local/bin

# Set the working directory in the container
WORKDIR /app

# Copy project files
COPY . /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the entrypoint command
CMD ["gunicorn", "--config", "/app/gunicorn_conf.py", "app:app"]
