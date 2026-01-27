FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    unzip \
    vim \
    tmux \
    git \
    build-essential \
    libglib2.0-dev \
    ninja-build \
    curl \
    wget \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/conda_setup

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda

ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --channel defaults && \
    conda tos accept --channel r

RUN conda install -y python=$PYTHON_VERSION && \
    conda clean -ya

RUN python -m pip install --upgrade pip ipython ipdb

RUN pip install attr

COPY . /opt/balrog/

WORKDIR /opt/balrog

RUN pip install --upgrade pip && \
    pip install -U attrs jsonschema && \
    pip install langchain==0.3.27 langchain-chroma==0.2.6 chromadb==1.3.0 finch-clust && \
    pip install . && \
    if [ -d "/opt/balrog/balrog/environments/nle" ]; then \
        mv /opt/balrog/balrog/environments/nle /opt/balrog/balrog/environments/nle_local; \
    fi && \
    balrog-post-install

CMD ["/bin/bash"]