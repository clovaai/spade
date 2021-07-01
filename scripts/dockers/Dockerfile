# 20201118 Wonseok Hwang
FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
        apt-get install -y \
        wget \
        zlib1g-dev \
        libssl-dev \
        curl\
        build-essential \
        cmake \
        git \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        vim \
        locales
RUN locale-gen en_US.UTF-8

# install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda.sh && \
	/bin/bash Miniconda.sh -b -p /opt/conda && \
	rm Miniconda.sh

ENV PATH /opt/conda/bin:$PATH
RUN ln -s /opt/conda/bin/pip /opt/conda/bin/pip3

# setup python
RUN conda install python=3.6

# install faiss (from faiss Dockerfile 
RUN conda install faiss-cpu -c pytorch

# install tensorflow, pytorch, keras
RUN pip3 install --upgrade pip
#RUN pip3 install --no-cache-dir tensorflow-gpu==2.2.0 torch==1.5.1 keras==2.4.0 torchvision numpy==1.17.2
RUN pip3 install --no-cache-dir tensorflow-gpu==1.12.0 keras==2.2.4 
RUN pip3 install numpy==1.17.2
RUN pip install torch==1.7.0+cu92 torchvision==0.8.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
ENV LANG C.UTF-8

# ====== Custom here =====
RUN pip3 install scipy==1.3.1 matplotlib==3.2.2 pymysql pandas==1.0.5 ipython==7.15.0 scikit-learn==0.23
RUN pip3 install gensim

# ====== Install apex ======
WORKDIR /tmp/unique_for_apex
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /tmp/unique_for_apex/apex
ENV TORCH_CUDA_ARCH_LIST 6.0;6.1;6.2;7.0
RUN python setup.py install --cpp_ext --cuda_ext
#RUN pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#WORKDIR /workspace

