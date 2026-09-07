# syntax=docker/dockerfile:1
ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1
FROM ${BASE_IMAGE}

USER root

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# 1. Install system prerequisites & ROS 2 Jazzy repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    git \
    unzip \
    build-essential \
    libboost-dev \
    python3-pip \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-base \
    ros-jazzy-geometry-msgs \
    ros-jazzy-nav-msgs \
    ros-jazzy-sensor-msgs \
    ros-jazzy-visualization-msgs \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-tf-transformations \
    ros-jazzy-nav2-behavior-tree \
    ros-jazzy-behaviortree-cpp \
    ros-jazzy-angles \
    ros-jazzy-nav2-util \
    ros-jazzy-nav2-msgs \
    ros-jazzy-resource-retriever \
    ros-jazzy-rviz2 \
    ros-jazzy-rviz-common \
    ros-jazzy-rviz-default-plugins \
    ros-jazzy-rviz-ogre-vendor \
    ros-jazzy-rviz-rendering \
    qtbase5-dev \
    python3-colcon-common-extensions \
    python3-yaml \
    python3-numpy \
    python3-matplotlib \
    python3-pandas \
    && rm -rf /var/lib/apt/lists/*

# 2. Install lightsfm (header-only Social Force Model lib required by
#    hunav_agent_manager; its CMakeLists looks for it under /usr/local/include)
RUN git clone --depth 1 https://github.com/robotics-upo/lightsfm.git /tmp/lightsfm \
    && mkdir -p /usr/local/include/lightsfm \
    && cp -r /tmp/lightsfm/include/* /usr/local/include/lightsfm/ \
    && rm -rf /tmp/lightsfm

# 3. Build HuNavSim packages (hunav_msgs & hunav_agent_manager) for Jazzy.
#    people_msgs has no Jazzy release, so it is built from source; the rest of
#    the wg-perception/people repo is ROS1-era OpenCV code and must be skipped.
WORKDIR /workspace/hunav_ws/src
RUN git clone --depth 1 https://github.com/robotics-upo/hunav_sim.git \
    && git clone --depth 1 -b ros2 https://github.com/wg-perception/people.git \
    && for p in people face_detector leg_detector people_tracking_filter people_velocity_tracker; do \
        touch people/$p/COLCON_IGNORE; done
# hunav_agent_manager includes ament_index_cpp and rcpputils headers but declares
# neither; Humble's rclcpp leaked their include dirs transitively, Jazzy's does not.
RUN cd hunav_sim/hunav_agent_manager \
    && sed -i \
        -e 's|^find_package(rclcpp REQUIRED)|find_package(rclcpp REQUIRED)\nfind_package(ament_index_cpp REQUIRED)\nfind_package(rcpputils REQUIRED)|' \
        -e 's|^ament_target_dependencies(hunav_agent_manager rclcpp|ament_target_dependencies(hunav_agent_manager rclcpp ament_index_cpp rcpputils|' \
        -e 's|^ament_target_dependencies(hunav_loader rclcpp|ament_target_dependencies(hunav_loader rclcpp ament_index_cpp rcpputils|' \
        CMakeLists.txt \
    && sed -i 's|<depend>rclcpp</depend>|<depend>rclcpp</depend>\n  <depend>ament_index_cpp</depend>\n  <depend>rcpputils</depend>|' package.xml

WORKDIR /workspace/hunav_ws
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    colcon build --packages-select people_msgs hunav_msgs hunav_agent_manager \
                                  hunav_evaluator hunav_behavior_tree_generator \
                                  hunav_rviz2_panel"

# 4. Setup Entrypoint script
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace/Hunav_isaac_wrapper

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
