# 1. Base Image: Lightweight Debian with Python 3.11
FROM python:3.11-slim

# 2. Prevent interactive apt prompts and configure Python caching
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 3. Install core system build tools and networking utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 4. Set container workspace
WORKDIR /workspace

# Create the virtual environment
ENV VIRTUAL_ENV=/opt/venv
# Activate venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV

# 5. Install requested data science, ML, LLM, and notebook libraries
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
# 6. Expose ports: 8888 for Jupyter, 5000 for MLflow UI
EXPOSE 8888 5000

# 8. Start interactive bash shell by default
CMD ["/bin/bash"]