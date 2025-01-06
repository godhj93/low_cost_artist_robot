docker run -it \
    --gpus all \
    --privileged \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env=NVIDIA_VISIBLE_DEVICES=all \
    --env=NVIDIA_DRIVER_CAPABILITIES=all \
    --net=host \
    --ipc=host \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v /dev:/dev \
    -v ~/:/server/ \
    --name low_cost_artist \
     low_cost_artist:latest \
    /bin/bash