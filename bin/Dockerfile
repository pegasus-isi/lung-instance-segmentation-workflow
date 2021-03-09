FROM opensciencegrid/tensorflow-gpu:2.3-cuda-10.1
WORKDIR /usr/bin
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa
RUN apt update && \
    apt install -y \
    python3.6 \
    python3-pip \
    libgl1-mesa-glx \
    wget \
    curl
RUN pip3 install --upgrade pip setuptools
RUN pip3 install \
    scikit-learn \
    h5py \
    numpy==1.18.4 \
    pandas \
    opencv-python \
    keras==2.3.1 \
    ray \
    ray[tune] \
    ray[rllib] \
    dataclasses \
    tabulate  \
    matplotlib
COPY preprocess.py ./
RUN chmod 777 ./preprocess.py
COPY hpo.py ./
RUN chmod 777 ./hpo.py
COPY unet.py ./
RUN chmod 777 ./unet.py
COPY train_model.py ./
RUN chmod 777 ./train_model.py
COPY prediction.py ./
RUN chmod 777 ./prediction.py
ENV LANG C.UTF-8
