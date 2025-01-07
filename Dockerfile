# Use the official Ubuntu Bionic as a parent image
FROM nvcr.io/nvidia/pytorch:23.04-py3

# Set maintainer label
LABEL maintainer="godhj@unist.ac.kr"

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Run package updates and install packages
RUN apt-get update \
    && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    nano \
    lsb-release \
    lsb-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add ROS repository and install ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update \
    && apt-get install -y ros-noetic-desktop-full

# Set up ROS environment and install additional tools
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash" \
    && apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

RUN apt-get install -y python3-rosdep

RUN rosdep init \
    && rosdep update

RUN apt install -y python3-catkin-tools

# Update pip
RUN pip install --upgrade pip

# Install additional packages for Stable Diffusion
RUN pip install transformers diffusers==0.29.0 vtracer svgpathtools accelerate gtts pyaudio
RUN apt install -y python3-pyaudio

# Install Whisper
RUN pip install -U openai-whisper
RUN apt install -y mpg123 ffmpeg

# Install MuJoCo
RUN pip install mujoco termcolor dynamixel_sdk

# Set the working directory in the container
WORKDIR /root/

RUN git clone https://github.com/godhj93/low_cost_artist_robot.git