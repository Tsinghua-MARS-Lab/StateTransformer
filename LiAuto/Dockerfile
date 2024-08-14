# config Pytorch and CUDA version

ARG PYTORCH="2.2.1"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# change source to speed up
RUN sed -i "s/archive.ubuntu./mirrors.aliyun./g" /etc/apt/sources.list && \
    sed -i "s/deb.debian.org/mirrors.aliyun.com/g" /etc/apt/sources.list && \
    sed -i "s/security.debian.org/mirrors.aliyun.com\/debian-security/g" /etc/apt/sources.list

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 openssh-server openssh-client vim tmux\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# For torch & CUDA optimization
ENV TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"

# Download miniconda and install silently.
ENV PATH /opt/conda/bin:$PATH
CMD curl -fsSLo Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    conda clean -a -y && \
 # Change conda and pip source to speed up
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
 conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
 conda config --set show_channel_urls yes && \
 pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
\
# fix ssh PATH
 echo 'export PATH=/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/games:/usr/local/games:/snap/bin:$PATH' | tee -a ~/.bashrc && \
\
# Install Dependencies for our State Transformer
## Install PyTorch (Newest Stable Version)
 pip3 install torch torchvision torchaudio && \
#  Install Transformers and related packages
 pip install transformers accelerate datasets numpy scikit-learn deepspeed ninja wandb opencv-contrib-python && \
 pip install flash-attn --no-build-isolation

# Install NuPlan-Devkit
CMD git clone https://github.com/motional/nuplan-devkit.git && \
    cd nuplan-devkit && \
    pip install -e . && \
    pip install aioboto3 && \
    pip install retry && \
    pip install aiofiles && \
    pip install bokeh==2.4.1
