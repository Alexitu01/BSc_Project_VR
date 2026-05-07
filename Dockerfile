FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

# System deps - ffmpeg and libgl
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    software-properties-common build-essential wget curl git ca-certificates \
    ffmpeg libgl1 libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir /workspace
WORKDIR /workspace

# PyTorch — cu121 as specified in README
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121 && \
    # plyfile - listed as optional in README but needed for .ply export
    pip install --no-cache-dir plyfile


COPY . .

#Dependencies - first: runpod
RUN pip install --no-cache-dir runpod && \
    #google-auth dependencies
    pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client && \
    #Pillow installation
    pip install pillow && \
    pip install -r requirements.txt

RUN git submodule update --init --recursive

WORKDIR /workspace/Projection/ml-sharp

RUN wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt

WORKDIR /workspace/Projection/DA360

RUN mkdir -p checkpoints && \
    pip install gdown && \
    bash scripts/download_models.sh

WORKDIR /workspace

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
    
ENV PATH="/opt/conda/bin:$PATH"

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -y -n sharp python=3.11
RUN conda run -n sharp pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN conda run -n sharp pip install -e /workspace/Projection/ml-sharp
RUN conda run -n sharp pip install -r requirements-sharp.txt

WORKDIR /workspace/Projection


RUN conda create -y -n da360 python=3.10
RUN conda run -n da360 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN conda run -n da360 pip install -r /workspace/requirements-da360.txt

# Model weights download at runtime via startup.py
# keeps the image lean and puts weights on the persistent volume
ENV HF_HOME="/runpod-volume/huggingface/data_Cache"
ENV HF_HUB_CACHE="/runpod-volume/huggingface/model_Cache"
ENV TMPDIR="/runpod-volume/tmp"

RUN mkdir -p /runpod-volume/huggingface/data_Cache \
             /runpod-volume/huggingface/model_Cache \
             /runpod-volume/tmp

VOLUME ["/runpod-volume"]

CMD ["python3.11", "-u", "runpodhandler.py"]
