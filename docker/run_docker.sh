#!/bin/bash

# IMAGE_NAME=yoshi/flow_field_based_motion_planner:latest
# IMAGE_NAME=yoshi/ffmp1206:latest
# IMAGE_NAME=ffmp0105:latest
IMAGE_NAME=5aa04ce6ea52

DIR=$(cd $(dirname $0); pwd)
# docker run -it --rm -p 10000:20022\
docker run -it --rm -p 6006:6006\
  --gpus all \
  --privileged \
  --env=QT_X11_NO_MITSHM=1 \
  --env=DISPLAY=$DISPLAY \
  --volume="/etc/group:/etc/group:ro" \
  --volume="/etc/passwd:/etc/passwd:ro" \
  --volume="/etc/shadow:/etc/shadow:ro" \
  --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/dev:/dev" \
  --volume="$PWD/../:/root/catkin_ws/src/flow_field_based_motion_planner" \
  --volume="$DIR/.ros/:/root/.ros/" \
  --net="host" \
  --name="ffmp" \
  $IMAGE_NAME \
  bash
