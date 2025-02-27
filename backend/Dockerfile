FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install dependencies# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    --allow-change-held-packages \
    build-essential \
    cmake \
    ccache \
    software-properties-common \
    git \
    curl \
    wget \
    sudo \
    vim \
    htop \
    tmux \
    # Python and pip
    python3 \
    python3-pip \
    python3-setuptools \
    python3-dev \
    python3-venv \
    # Multimedia and visualization libraries
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libfontconfig1 \
    libopenexr25 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    # CUDA and NCCL libraries
    libnccl2 \
    libnccl-dev \
    cuda-toolkit-12-4 \
    cuda-libraries-12-4 \
    cuda-nvtx-12-4 \
    cuda-nvml-dev-12-4 \
    cuda-command-line-tools-12-4 \
    cuda-cupti-dev-12-4 \
    cuda-minimal-build-12-4 \
    && rm -rf /var/lib/apt/lists/*
    
# Set up Python and pip
RUN python3 -m pip install --upgrade pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /usr/src/app
WORKDIR /usr/src/app

# Expose the port your app runs on
EXPOSE 7861
EXPOSE 1370

# Command to run your application
CMD ["python3", "app.py"]