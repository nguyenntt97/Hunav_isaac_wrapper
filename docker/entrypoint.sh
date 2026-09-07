#!/bin/bash
set -e

# 1. Source ROS 2 Jazzy and HuNavSim workspace
source /opt/ros/jazzy/setup.bash
if [ -f "/workspace/hunav_ws/install/setup.bash" ]; then
    source /workspace/hunav_ws/install/setup.bash
fi

# 2. Auto-extract Carter USD if not already extracted
CARTER_ZIP="/workspace/Hunav_isaac_wrapper/src/config/robots/nova_carter_ros2_sensors.zip"
CARTER_USD="/workspace/Hunav_isaac_wrapper/src/config/robots/nova_carter_ros2_sensors.usd"

if [ -f "$CARTER_ZIP" ] && [ ! -f "$CARTER_USD" ]; then
    echo ">> Extracting Carter robot USD asset..."
    unzip -q -o "$CARTER_ZIP" -d "/workspace/Hunav_isaac_wrapper/src/config/robots/"
fi

# 3. Expose the wrapper to colcon. The repo is bind-mounted at run time, so the
#    link cannot be made during the image build. Nothing needs to be built for
#    ./launch_hunav_isaac.sh to work -- this only enables the optional
#    `colcon build --packages-select hunav_isaac_wrapper`, which in turn enables
#    `ros2 run hunav_isaac_wrapper hunav_isaac_launcher`.
WRAPPER_SRC="/workspace/Hunav_isaac_wrapper/src"
WRAPPER_LINK="/workspace/hunav_ws/src/hunav_isaac_wrapper"
if [ -d "$WRAPPER_SRC" ] && [ ! -e "$WRAPPER_LINK" ]; then
    ln -s "$WRAPPER_SRC" "$WRAPPER_LINK"
fi

# 4. Add Isaac Sim python directory to PATH
if [ -d "/isaac-sim" ]; then
    export PATH="/isaac-sim:$PATH"
fi

exec "$@"
