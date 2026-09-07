# **HuNav Isaac Wrapper**  

A standalone simulation wrapper for **NVIDIA Isaac Sim**, integrating the **Human Navigation Simulator ([HuNavSim](https://github.com/robotics-upo/hunav_sim))** with **physics-based animations** and **ROS 2 integration**.

---

### 🚧 **Work in Progress**

This repository is actively developed and subject to improvements.

### ✅ **Tested Configurations**  

- **ROS 2 Humble** + **Isaac Sim 4.5** + **Ubuntu 22.04 LTS**  
- **ROS 2 Jazzy** + **Isaac Sim 6.0** + **Ubuntu 24.04 LTS** (see `docker/`)  

Isaac Sim 5.0 reorganised the asset buckets and moved several extensions out of the
default app. Version-sensitive asset paths are resolved at runtime by
`src/hunav_isaac_wrapper/asset_paths.py`; the required animation extensions are enabled
at runtime by `hunav_manager.py`. No manual Isaac Sim configuration is needed.

## 🔹 **Overview**  

**HuNav Isaac Wrapper** is a modular simulation framework that integrates the [HuNavSim](https://github.com/robotics-upo/hunav_sim) human navigation simulator into **NVIDIA Isaac Sim**, enabling realistic multi-agent behavior with physics-based animation and **ROS 2** interoperability.

It supports both **ROS 2 teleoperation** and **autonomous navigation (Nav2)**, world loading, dynamic agent configuration, and multiple robot models.

*This wrapper is built for research in **human-robot interaction**, **social navigation**, and **simulation-based validation of social navigation policies**.*

---

## 🔹 **Features**  

- **ROS2 Workspace Structure:**
  - Complete ROS2 workspace that can be cloned and built directly with `colcon build`

- **Modular Architecture:**  
  - `main.py`: Interactive launcher providing a command-line interface for configuration selection (agent files, worlds, robots) and simulation startup
  - `world_builder.py`: Loads USD world files
  - `hunav_manager.py`: Handles agent creation, communication with HuNavSim services, and manages physics, animations, and obstacle detection
  - `teleop_hunav_sim.py`: Manages HuNavSim initialization, updates agent states, and handles the ROS 2 /cmd_vel interface for robot control
  - `animation_utils.py`: Utilities for AnimationGraph setup and retargeting

- **Enhanced Agent Configuration:**  
  - YAML files (in `scenarios/`) define agent spawn positions, navigation goals, SFM parameters, and behavior profiles
  - Backward compatible with existing configuration files

- **Animation System:**  
  - **AnimationGraph-based** blending for smooth walk/idle transitions  
  - Driven by agent velocity, allowing dynamic switching between walk and idle states
  - Supports animation **retargeting**, applying a single set of animations to different characters via **USD SkelAnimation** and the **Omni Anim Retargeting extension**

- **Multiple Robot Models:**  
  - Includes `jetbot`, `create3`, `carter`, and `carter_ROS` models

- **ROS 2 Navigation (Nav2) Support:**  
  - Enables autonomous navigation for the **Carter** robot using the **ROS 2 Nav2** stack

- **Social Simulation & Teleoperation:**  
  - Real-time robot control via `/cmd_vel`  
  - Socially-aware agent movement via **HuNavSim** integration

- **Obstacle Detection:**  
  - Uses **PhysX raycasts** (in `hunav_manager.py`) for detecting obstacles and informing **HuNavSim** navigation logic

---

## 🔹 **Requirements**

- **Ubuntu 22.04 LTS** or **24.04 LTS**
- [**HuNavSim**](https://github.com/robotics-upo/hunav_sim)
- [**NVIDIA Isaac Sim (Workstation Installation)**](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html) — 4.5 or 5.0+

- **Python 3.8+**  

- **ROS 2** [Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) or [Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html)  

- Network access to the Isaac Sim asset bucket (characters, animations and the
  retargeting source skeleton are streamed, not bundled).

---

## 🔹 **Setup Guide**

### 1. Repository Setup

#### **Option 1: Direct Clone (Recommended)**

This project is structured as a complete ROS2 workspace:

```bash
# Clone the repository
git clone https://github.com/robotics-upo/Hunav_isaac_wrapper
cd Hunav_isaac_wrapper

# Install ROS2 dependencies
sudo apt install ros-humble-geometry-msgs ros-humble-nav-msgs ros-humble-sensor-msgs ros-humble-tf2-ros

# Install Python dependencies
pip install pyyaml numpy matplotlib

# Build the workspace 
colcon build

# Source the workspace
source install/setup.bash
```

#### **Option 2: Add to Existing ROS2 Workspace**

If you want to integrate this into an existing ROS2 workspace:

```bash
cd ~/your_ros2_ws/src
git clone https://github.com/robotics-upo/Hunav_isaac_wrapper.git

# Move the package contents to your workspace
cp -r hunav_isaac_wrapper/src/* .
rm -rf hunav_isaac_wrapper

# Build your workspace
cd ~/your_ros2_ws
colcon build --packages-select hunav_isaac_wrapper
source install/setup.bash
```

#### **Additional Dependencies**

You'll also need to install HuNavSim in case you haven't already:

```bash
# In your ROS2 workspace src directory
git clone https://github.com/robotics-upo/hunav_sim.git
cd .. && colcon build 
```

### 2. Isaac Sim Installation

Make sure you have **NVIDIA Isaac Sim** installed. Follow the [Isaac Sim Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).

### 3. Isaac Sim Extensions

Nothing to do. `hunav_manager.py` enables `omni.anim.retarget.core` and
`omni.anim.graph.core` at startup via `enable_extension()`.

⚠️ Earlier revisions of this README told you to copy `src/isaacsim.exp.base.kit` over the
one in your Isaac Sim installation. **Do not do this.** That file targets Isaac Sim 4.5;
31 of its 119 extension dependencies (the whole legacy `omni.isaac.*` family,
`isaacsim.asset.browser`, `omni.kit.loop-isaac`, …) do not exist in 5.0+, and copying it
prevents the app from starting.

### 4. ROS 2 Setup

Ensure ROS 2 is installed and sourced:

```bash
source /opt/ros/jazzy/setup.bash    # or /opt/ros/humble/setup.bash
```

### 5. Configure Your Scene, Agents, and Robot

The simulation setup is configured through the interactive launcher, which provides a menu-driven interface to select the world, agent configuration, and robot model.

#### 🗺️ Scene Setup

The repository includes multiple pre-configured USD files in `src/worlds/`:

- `warehouse.usd`: Industrial layout with shelves and various obstacles
- `hospital.usd`: Medical environment with corridors and rooms
- `office.usd`: Workspace with desks and conference areas
- `empty_world.usd`: A minimal open environment for testing

#### 🧍 Agents Configuration

Each world is paired with a YAML file in `src/scenarios/` that defines the HuNavSim agents:

- **Available configuration files:**
  - `agents_warehouse.yaml` → for `warehouse.usd`
  - `agents_hospital.yaml` → for `hospital.usd`

- **Each YAML file lets you configure:**
  - **Initial pose**: Define starting positions of each agent.
  - **Goals**: Set destination coordinates or waypoints per agent.
  - **SFM weights**: Tune the social force model for realistic crowd behavior.
  - **Behavior type**: Choose how agents behave.

**Always pair the world with its corresponding agent YAML to avoid misaligned goals or initial agent positions**.

#### 🤖 Robot Configuration

The interactive launcher will prompt you to select your desired robot:
- `jetbot`, `create3`, `carter`, or `carter_ROS`

**Note:** For `carter_ROS`, make sure to unzip the `nova_carter_ros2_sensors` package located in `src/config/robots/`. (The Docker entrypoint does this for you.)

**Note:** `carter` is **Isaac Sim 4.5 only**. It used `Isaac/Robots/Carter/nova_carter_sensors.usd`,
which has no sensor-equipped equivalent in the 5.0+ asset buckets. On 5.0+ use `carter_ROS`,
which is backed by the USD bundled in this repository; selecting `carter` there fails with
an explicit error rather than a 404.

**Carter** robot also supports **ROS 2 Navigation (Nav2)** for autonomous navigation.

### 6. Launch the Simulation

You can launch the simulation using two methods:

#### Method 1: Shell Script

```bash
cd ~/Hunav_isaac_wrapper  # Navigate to the repository root
# Make sure the script is executable
chmod +x launch_hunav_isaac.sh
# Launch the simulation
./launch_hunav_isaac.sh
```

#### Method 2: ROS2 Run Command

```bash
# Interactive mode
ros2 run hunav_isaac_wrapper hunav_isaac_launcher

# With custom arguments
ros2 run hunav_isaac_wrapper hunav_isaac_launcher --config warehouse_agents.yaml --robot carter_ROS --batch
```

Both methods provide the same functionality and will:

- Start the interactive launcher with menu-driven configuration (when no scenario specified)
- Guide you through selecting agent configuration and robot model
- Load the specified world and spawn agents based on your selections
- Initialize physics, animations, and ROS 2 integration

### 7. Teleoperation and Navigation

#### Teleoperation

Use ROS 2 to publish Twist messages to `/cmd_vel` for direct robot control:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: 0.1}}'
```

#### ROS 2 Navigation (Nav2) – Carter Robot Only

For autonomous navigation with the `carter_ROS` robot:

1. Ensure the simulation is running
2. In a separate terminal, launch the navigation stack:

   ```bash
   ros2 launch carter_navigation carter_navigation.launch.py \
     params_file:="src/config/navigation_params/carter_navigation_params.yaml" \
     map:="src/maps/warehouse.yaml"
   ```

## 🔹 **Troubleshooting**
  
### Automated Setup Issues

If you encounter issues with the manual setup steps, you can use the automated setup script:

  ```bash
  # Make the script executable
  chmod +x setup_workspace.sh

  # Run the setup script
  ./setup_workspace.sh
  ```

This script will:

- Verify ROS2 is properly sourced
- Check for required dependencies (`hunav_msgs`, `geometry_msgs`, `std_msgs`, `nav_msgs`, `sensor_msgs`, `tf2_ros`)
- Automatically build the package if in a colcon workspace
- Provide guidance for workspace creation if needed
- Display usage examples for different launch methods

### ROS 2 Connectivity

Make sure that your ROS 2 Humble installation is sourced:

  ```bash
  source /opt/ros/humble/setup.bash
  ```

If `carter_navigation` package is not recognized, follow these steps:

1. Clone the [IsaacSim-ros_workspaces](https://github.com/isaac-sim/IsaacSim-ros_workspaces.git) repository:

   ```bash
   git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git
   ```

2. Build the ROS 2 humble workspace:

   ```bash
   cd IsaacSim-ros_workspaces/humble_ws
   colcon build
   ```

3. Source the workspace in your `.bashrc`:

   ```bash
   source ~/IsaacSim-ros_workspaces/humble_ws/install/setup.bash
   ```

### Agent Configuration Issues

- If agents appear horizontal rather than vertical, adjust the rotation applied in the script (e.g., modify the quaternion calculation in `hunav_manager.initialize_agents()`'s `init_rot` parameter and/or `hunav_manager._update_agents()`).

### Retargeting Errors (*)

- Verify that the default source biped prim is correctly loaded at `/World/biped_demo`.
- Ensure that the target agents have a valid **skeleton** (use the `findSkeletonPath` method for debugging).
- Confirm that the extension `omni.anim.retarget.core` is enabled.

### AnimationGraph Issues

- If AnimationGraph is not applying correctly, verify that it is correctly assigned to the agent's **SkelRoot** and that transformations are applied properly.

---

## Acknowledgments

This work is carried out as part of the **HunavSim 2.0** project, _“A Human Navigation Simulator for Benchmarking Human-Aware Robot Navigation”_, supported under the [**euROBIN 2nd Open Call – Technology Exchange Programme**](https://www.eurobin-project.eu/index.php/showroom/news/47-2nd-call-eurobin-technology-exchange-programme) (**euROBIN_2OC_2**), funded by the **European Union's Horizon Europe** research and innovation programme under grant agreement **No. 101070596**.

<p align="left">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/84/European_Commission.svg/300px-European_Commission.svg.png" width="160"/>
  &nbsp;&nbsp;&nbsp;
  <img src="https://www.eurobin-project.eu/images/2025/03/15/eurobin_logo-_payoff.png" alt="euROBIN Logo" width="160"/>
</p>

