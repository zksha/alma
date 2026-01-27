FROM continuumio/miniconda3:latest

WORKDIR /opt/evals

RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    gcc \
    zlib1g-dev \
    libffi-dev \
    libncurses5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml ./
COPY requirements.txt ./
RUN conda env create -f environment.yml && conda clean -afy

SHELL ["conda", "run", "-n", "alfworld-env", "/bin/bash", "-c"]

CMD ["bash"]

