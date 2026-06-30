FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-imgproc-dev \
    netcat-openbsd \
    python3-catkin-tools \
    ros-noetic-geometry-msgs \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    ros-noetic-roscpp \
    ros-noetic-std-msgs \
    tmux \
    xauth \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

CMD ["bash", "-lc", "trap : TERM INT; sleep infinity & wait"]
