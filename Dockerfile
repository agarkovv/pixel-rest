ARG BASE_CONTAINER=ubuntu
ARG UBUNTU_VERSION=22.04

FROM $BASE_CONTAINER:$UBUNTU_VERSION

ARG PYTHON_VERSION=3.11
ARG CONDA_ENV=pixel-rest

LABEL version="0.0.1"

ENV WORKDIR=/workspace/pixel-rest
WORKDIR $WORKDIR

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    wget \
    curl \
    git \
    vim \
    openssh-client \
    build-essential \
    zip \
    unzip \
    ffmpeg \
    libsm6 \
    libxext6

# install Poetry
# following https://python-poetry.org/docs/#ci-recommendations
ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

RUN python3.11 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==$POETRY_VERSION

ENV PATH="${PATH}:${POETRY_VENV}/bin"
RUN poetry config virtualenvs.create false

# install conda
# same as https://hub.docker.com/r/continuumio/miniconda3
ENV CONDA_VERSION=py39_4.12.0
ENV CONDA_PATH=/opt/conda

RUN export UNAME_M="$(uname -m)" \
    && wget -O /tmp/miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-${UNAME_M}.sh \
    && bash /tmp/miniconda3.sh -b -p $CONDA_PATH
RUN ln -s ${CONDA_PATH}/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". ${CONDA_PATH}/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && find ${CONDA_PATH}/ -follow -type f -name '*.a' -delete \
    && find ${CONDA_PATH}/ -follow -type f -name '*.js.map' -delete \
    && ${CONDA_PATH}/bin/conda clean -afy

SHELL ["/bin/bash", "--login", "-c"]

# Mount dataset, not copy. TODO: add cloud mounting
ADD data $WORKDIR/data

# Copy project to the docker, excluding data dir
COPY . $WORKDIR

# Create venv and install project dependencies
RUN cd $WORKDIR \
    && conda create -n $CONDA_ENV python=$PYTHON_VERSION \
    && conda run -n $CONDA_ENV poetry install