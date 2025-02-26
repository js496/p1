FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    --allow-change-held-packages \
    ccache \
    software-properties-common \
    git \
    curl \
    wget \
    sudo \
    vim \
    python3-pip \
    python3-setuptools \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Python and pip
RUN python3 -m pip install --upgrade pip

# Install additional Python packages
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Expose the port your app runs on
EXPOSE 7861
EXPOSE 1370

# Command to run your application
CMD ["python3", "app.py"]