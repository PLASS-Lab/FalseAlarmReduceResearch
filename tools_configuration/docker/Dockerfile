FROM ubuntu:latest

## for apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

# RUN apt-get update && \
#   apt-get install -y sudo \
#   curl \
#   wget \
#   gnupg \
#   zsh \
#   git-core \
#   git \
#   linuxbrew-wrapper \
#   fonts-powerline \
#   locales \
#   nano \
#   vim \
#   nodejs \
#   npm \
#   python3 \
#   matplotlib \
#   net-tools \
#   iputils-ping \
#   openssh-server 

RUN apt-get update &&  apt-get install -y sudo
RUN apt-get -y install zsh
RUN apt-get -y install iputils-ping
RUN apt-get -y install python3
RUN apt-get -y install curl wget
RUN apt-get -y install git vim 
RUN apt-get -y install net-tools 
RUN apt-get -y install openssh-server
RUN apt-get -y install cppcheck flawfinder


# GIT config 
# RUN git config --global user.name huonghope
# RUN git config --global user.email hopekr@dongguk.edu

# SSHD setup
RUN mkdir /var/run/sshd
RUN echo 'root:1997' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN mkdir /home/hopekr
WORKDIR /home/hopekr

CMD [ "zsh" ]
EXPOSE 22