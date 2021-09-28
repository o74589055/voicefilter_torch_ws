# syntax = docker/dockerfile:experimental
FROM nvidia/cuda:11.1-devel-ubuntu20.04

# Arguments
ARG USER=initial
ARG GROUP=initial
ARG UID=1000
ARG GID=${UID}
ARG SHELL=/bin/bash

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Setup users and groups
RUN groupadd --gid ${GID} ${GROUP} \
  && useradd --gid ${GID} --uid ${UID} -ms ${SHELL} ${USER} \
  && mkdir -p /etc/sudoers.d \
  && echo "${USER}:x:${UID}:${UID}:${USER},,,:$HOME:${shell}" >> /etc/passwd \
  && echo "${USER}:x:${UID}:" >> /etc/group \
  && echo "${USER} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${USER}" \
  && chmod 0440 "/etc/sudoers.d/${USER}"
# Replace apt urls
RUN sed -i 's@archive.ubuntu.com@ftp.tku.edu.tw/@g' /etc/apt/sources.list

# Install packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \  
  apt-get install -y --no-install-recommends \
  sudo net-tools htop git rsync wget zip unzip \
  byobu zsh \
  nano vim \
  terminator gnome-terminal libcanberra-gtk3-module \
  dbus-x11 libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 \
  ca-certificates \
  libjpeg-dev \
  libpng-dev \
  python3-pip \
  python3-setuptools \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > \
#       /etc/apt/sources.list.d/ros-latest.list \
#       && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' \
#       --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
#       && curl -sSL \
#       'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' \
#       | sudo apt-key add - \
#       && apt update && apt install -y --no-install-recommends ros-noetic-desktop-full \
#       && apt-get clean \
#       && rm -rf /var/lib/apt/lists/*

# ENV ROS_DISTRO noetic

WORKDIR /home/${USER}/.tmp

# setup custom configuration
COPY config .
# shell
RUN bash ./script/bash_setup.bash \
  && bash ./script/terminator_setup.bash \
  && bash ./script/zsh_setup.bash \
  && bash ./script/byobu_setup.bash \
  # Dependent
  && bash ./script/pip_setup.bash \
  && rm -rf /home/${USER}/.tmp

# RUN pigpiod
WORKDIR /home/${USER}

# Switch user to ${USER}
USER ${USER}

RUN sudo mkdir work

# Make SSH available
EXPOSE 22

# Switch to user's HOME folder
WORKDIR /home/${USER}/work

# ENTRYPOINT ["/entrypoint.bash","byobu"]
ENTRYPOINT ["byobu"]
#  ENTRYPOINT ["bash"]
# ENTRYPOINT ["terminator"]
# ENTRYPOINT ["zsh"]
