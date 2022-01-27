ARG CUDA=11.1
FROM nvidia/cuda:${CUDA}-runtime-ubuntu18.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

# Update CUDNN to 8.2.0

ENV CUDNN_VERSION=8.2.0.53
RUN apt-get update && apt-get install -y --no-install-recommends     libcudnn8=$CUDNN_VERSION-1+cuda11.3     && apt-mark hold libcudnn8 &&     rm -rf /var/lib/apt/lists/* # buildkit

# Install apt-get packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      cmake \
      cuda-command-line-tools-${CUDA/./-} \
      git \
      tzdata \
      wget \
      cuda-compat-11-4 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install conda packages.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda update -qy conda \
    && conda install -y -c conda-forge \
      cudatoolkit==${CUDA_VERSION} \
      pip \
      python=3.7

COPY . /app/projax
# Install pip packages.
RUN pip3 install --upgrade pip \
    && pip3 install -r /app/projax/requirements.txt \
# Below versions works for 450 drivers.
#    && pip3 install jax==0.2.13 jaxlib==0.1.69+cuda${CUDA/./} -f \
#      https://storage.googleapis.com/jax-releases/jax_releases.html \
# For GPU driver > 455 please use below for more recent jax
    && pip3 install jax[cuda] -f \
      https://storage.googleapis.com/jax-releases/jax_releases.html \
    && pip3 install dm-haiku optax

ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.4/compat:$LD_LIBRARY_PATH

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.
WORKDIR /app/projax
RUN echo $'#!/bin/bash\n\
ldconfig\n\
python /app/projax/NeuralODE/train_ode_hk.py "$@"' > /app/run_ode_mnist.sh \
    && chmod +x /app/run_ode_mnist.sh

CMD ["/app/run_ode_mnist.sh"]
