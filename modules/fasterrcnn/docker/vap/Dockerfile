FROM pytorch/pytorch:latest
ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         vim \
         curl \
         ca-certificates \
         libjpeg-dev \
         libglib2.0-0 \ 
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN pip install torchvision opencv-python-headless matplotlib cython
RUN pip install pycocotools pandas
