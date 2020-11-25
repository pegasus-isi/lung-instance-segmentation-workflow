FROM ubuntu:18.04
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
    tensorflow==2.2.0 \
    h5py \
    numpy==1.18.4 \
    pandas \
    opencv-python \
    keras==2.3.1 \
    ray \
    ray[tune] \
    ray[rllib] \
    dataclasses \
    tabulate
COPY preprocess.py ./
RUN chmod 777 ./preprocess.py
COPY data_split.py ./
RUN chmod 777 ./data_split.py
COPY train_model.py ./
RUN chmod 777 ./train_model.py
RUN groupadd --gid 808 scitech-group
RUN useradd --gid 808 --uid 550 --create-home --password '$6$ouJkMasm5X8E4Aye$QTFH2cHk4b8/TmzAcCxbTz7Y84xyNFs.gqm/HWEykdngmOgELums1qOi3e6r8Z.j7GEA9bObS/2pTN1WArGNf0' riyuzaki
ENV LANG C.UTF-8
