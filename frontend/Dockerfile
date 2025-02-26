FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ccache software-properties-common git curl wget sudo vim python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . /usr/src/app
WORKDIR /usr/src/app

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python3", "app.py"]



