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

# 5. Install PyTorch CPU-only wheels directly
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install requested data science, ML, LLM, and notebook libraries
RUN pip install \
    numpy \
    matplotlib \
    jupyter \
    notebook \
    ipykernel \
    "lightning>=2.4.0" \
    "langchain>=0.3.0" \
    "langchain-community" \
    "optuna>=3.6.0" \
    "mlflow>=2.17.0" \
    "pandas>=2.0.0" \
    "langchain-google-genai>=0.1.0"
# 7. Expose ports: 8888 for Jupyter, 5000 for MLflow UI
EXPOSE 8888 5000

# 8. Start interactive bash shell by default
CMD ["/bin/bash"]