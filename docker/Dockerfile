FROM vistart/cuda:10.2-cudnn7-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive


# [0] ssh setup
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:screencast' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# [1] ROS install
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
		       ros-noetic-desktop-full


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                       wget \
                       python3-pip \
                       tk-dev \
                       python3-tk \
    && rm -rf /var/lib/apt/lists/*

#RUN source /opt/ros/noetic/setup.bash
#RUN export ROS_PACKAGE_PATH=/root/catkin_ws
#RUN export ROS_WORKSPACE=/root/catkin_ws
#RUN source /root/catkin_ws/devel/setup.bash && \


# [2] PyTorch install
RUN apt-get update && \
	apt-get install -y python3-setuptools
RUN pip3 install torch torchvision


# [3] X window and window manager
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        xvfb \
        x11vnc \
        python-opengl \
        icewm \
	build-essential \
	net-tools


# [4] install somethings for drl
RUN pip3 install gym
RUN pip3 install kornia
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    zlib1g-dev
    libjpeg-dev
    xvfb
    ffmpeg
    xorg-dev
    python-opengl
    libboost-all-dev
    libsdl2-dev
    swig


# vim install
RUN apt-get update && \
    apt-get install -y vim


# clean cache
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*



WORKDIR /root
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
