# Use an official lightweight Python image
FROM python:3.10-slim

# Create a non-root user with UID 1000 (Hugging Face Spaces runs as user 1000)
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Set the working directory
WORKDIR $HOME/app

# Copy configuration and project files
COPY --chown=user pyproject.toml README.md ./

# Install dependencies including search, training (AutoGluon), and API (FastAPI) groups
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -e ".[search,train,api]"

# Copy package source code
COPY --chown=user autohf ./autohf

# Hugging Face Spaces expects the app to listen on port 7860 by default
EXPOSE 7860

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "autohf.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
