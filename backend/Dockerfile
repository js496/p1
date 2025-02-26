FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ccache software-properties-common git curl wget sudo vim python3-pip \
    python3-setuptools ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /usr/src/app
WORKDIR /usr/src/app

# Expose the port your app runs on
EXPOSE 7861
EXPOSE 1370

# Command to run your application
CMD ["python3", "app.py"]