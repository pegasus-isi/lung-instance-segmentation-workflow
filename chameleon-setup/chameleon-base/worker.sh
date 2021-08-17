#!/bin/bash

# this script should run as root

apt-get update && apt-get -y upgrade
apt-get install -y linux-headers-$(uname -r)
apt-get install -y build-essential make zlib1g-dev librrd-dev libpcap-dev autoconf libarchive-dev iperf3 htop bmon vim wget pkg-config git python-dev python-pip libtool
pip install --upgrade pip

######################
### EDIT /etc/hosts ##
######################

cat << EOF >> /etc/hosts
10.100.101.217 panorama-data
EOF

######################
### INSTALL CONDOR ###
######################

wget -qO - https://research.cs.wisc.edu/htcondor/ubuntu/HTCondor-Release.gpg.key | sudo apt-key add -
echo "deb http://research.cs.wisc.edu/htcondor/ubuntu/8.8/bionic bionic contrib" >> /etc/apt/sources.list
echo "deb-src http://research.cs.wisc.edu/htcondor/ubuntu/8.8/bionic bionic contrib" >> /etc/apt/sources.list

apt-get update && apt-get install -y htcondor

cat << EOF > /etc/condor/config.d/50-main.config
use feature : GPUs

DAEMON_LIST = MASTER, STARTD

CONDOR_HOST = panorama-data

USE_SHARED_PORT = TRUE

NETWORK_INTERFACE = 10.100.100.*

# the nodes have shared filesystem
UID_DOMAIN = \$(CONDOR_HOST)
TRUST_UID_DOMAIN = TRUE
FILESYSTEM_DOMAIN = \$(FULL_HOSTNAME)

#--     Authentication settings
SEC_PASSWORD_FILE = /etc/condor/pool_password
SEC_DEFAULT_AUTHENTICATION = REQUIRED
SEC_DEFAULT_AUTHENTICATION_METHODS = FS,PASSWORD
SEC_READ_AUTHENTICATION = OPTIONAL
SEC_CLIENT_AUTHENTICATION = OPTIONAL
SEC_ENABLE_MATCH_PASSWORD_AUTHENTICATION = TRUE
DENY_WRITE = anonymous@*
DENY_ADMINISTRATOR = anonymous@*
DENY_DAEMON = anonymous@*
DENY_NEGOTIATOR = anonymous@*
DENY_CLIENT = anonymous@*

#--     Privacy settings
SEC_DEFAULT_ENCRYPTION = OPTIONAL
SEC_DEFAULT_INTEGRITY = REQUIRED
SEC_READ_INTEGRITY = OPTIONAL
SEC_CLIENT_INTEGRITY = OPTIONAL
SEC_READ_ENCRYPTION = OPTIONAL
SEC_CLIENT_ENCRYPTION = OPTIONAL

#-- With strong security, do not use IP based controls
HOSTALLOW_WRITE = *
ALLOW_NEGOTIATOR = *

# dynamic slots
SLOT_TYPE_1 = cpus=100%,disk=100%,swap=100%,gpus=100%
SLOT_TYPE_1_PARTITIONABLE = TRUE
NUM_SLOTS = 1
NUM_SLOTS_TYPE_1 = 1

EOF

condor_store_cred -f /etc/condor/pool_password -p c0nd0r_p00l

systemctl enable condor
systemctl restart condor

##########################
### INSTALL SINGULARITY ##
##########################

apt-get update && sudo apt-get install -y build-essential \
    uuid-dev \
    libgpgme-dev \
    squashfs-tools \
    libseccomp-dev \
    wget \
    pkg-config \
    git \
    cryptsetup-bin

export VERSION=1.16.2 OS=linux ARCH=amd64 && \
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz

echo 'export GOPATH=${HOME}/go' >> ~/.bashrc
echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> ~/.bashrc

export GOPATH=${HOME}/go >> ~/.bashrc
export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin >> ~/.bashrc

export VERSION=3.7.4 && # adjust this as necessary \
    wget https://github.com/hpcng/singularity/releases/download/v${VERSION}/singularity-${VERSION}.tar.gz && \
    tar -xzf singularity-${VERSION}.tar.gz && \
    rm singularity-${VERSION}.tar.gz && \
    cd singularity

./mconfig && \
    make -C ./builddir && \
    make -C ./builddir install

##########################
### INSTALL DOCKER      ##
##########################
cd
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -


apt-key fingerprint 0EBFCD88

add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io


#groupadd docker
usermod -aG docker condor

systemctl enable docker
systemctl restart docker

############################
### SETUP PANORAMA USER ####
############################
cd
useradd -s /bin/bash -d /home/panorama -m -G docker panorama

echo "panorama     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

mkdir /home/panorama/.ssh
chmod -R 700 /home/panorama/.ssh
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDMnR9XlDv/NiEyKPgMzO/WOcQ9ZoDt2BYC7CHB9EmJQG4dwtzhioLJspJ8t4IuHpw6JlxjybTYqVUJqbKKT56t7PdFrzy7R5D5MO31CcAMhzaaFL7mtviIj+jy4wEitZr5Jh7SGgPFTLA54cx3fHCsrs0I0PjSRcaKtEi0RK0HsmUNrh5cRFm1oOgShkthM9KMfZAJ2hhkaoneywGfBvfq3dOQkfFdTCxn3B+Sc28l6wtT+n9ruNhasQ3OqmkZ5lhg+/CH5zTd7dCy57Fd/BuFEUq3pdhLIXXhnxDTftn1Nwd6FLy865XlIMnSSt8ds/X3sndupkA7G5f6ZyDKZinJRjr+pGrKC5lly1L3sw/oPguQDfHJ7VJI/jVWP4A4Xp0etXw50pF0GgA9+UT84tBfe3LB4cMhdJ/UWrEgK/jjCtSIe9bahT4gCL2PIbIacOXFqla3DiEcw/ZcCr8hprFLey04BgDvbMN1x+AydXvLjl4eDar5/ey1AlLzaNLXobEdK17DMsG6I3spJFJ/MB18vEu+F4QpTh9A4btX81XFWssdXhynVrrSbMgepQQAYoa92VVAD/re9PgwMXDHaERJW190SyV+ruv0R9FEmp9izWN44tx8E6hyo/eHZ7H65DlBilRQBehsefN7dY0BApLAxmpRkuwa0c1XE0UkEkZQOw== georgpap@iris.isi.edu" >> /home/panorama/.ssh/authorized_keys
chmod 600 /home/panorama/.ssh/authorized_keys
chown -R panorama:panorama /home/panorama/.ssh

echo 'export GOPATH=${HOME}/go' >> /home/panorama/.bashrc
echo 'export PATH=/usr/local/go/bin:${PATH}:${GOPATH}/bin' >> /home/panorama/.bashrc

# final steps might be needed after restart
echo "10.100.101.217 panorama-data" >> /etc/hosts
apt-get install -y linux-headers-$(uname -r)
echo off > /sys/devices/system/cpu/smt/control
systemctl stop ufw
systemctl restart condor
