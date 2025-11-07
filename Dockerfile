FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11 and system dependencies
# Ubuntu 20.04 doesn't have Python 3.11 in default repos, so we use deadsnakes PPA
# If Python 3.11 isn't available, we'll build from source
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && (apt-get install -y python3.11 python3.11-dev python3.11-distutils || \
        (echo "Python 3.11 not available in PPA, building from source..." && \
         apt-get install -y wget make zlib1g-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev && \
         cd /tmp && \
         wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz && \
         tar xzf Python-3.11.9.tgz && \
         cd Python-3.11.9 && \
         ./configure --enable-optimizations --prefix=/usr/local && \
         make -j$(nproc) && \
         make altinstall && \
         ln -sf /usr/local/bin/python3.11 /usr/bin/python3.11)) \
    && apt-get install -y git build-essential \
    && rm -rf /var/lib/apt/lists/* /tmp/Python-3.11.9*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    python --version

# Set working directory
WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set Python path
ENV PYTHONPATH=/workspace

# Default command: keep container running for interactive use
CMD ["tail", "-f", "/dev/null"]

