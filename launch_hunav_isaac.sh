#!/bin/bash
#
# launch_hunav_isaac.sh
#
# Simple launcher script that mimics the original usage:
# bash ~/isaacsim/python.sh ~/Hunav_isaac_wrapper/main.py
#
# This script can be used in three ways:
# 1. Interactive mode: ./launch_hunav_isaac.sh
# 2. With scenario: ./launch_hunav_isaac.sh warehouse_agents.yaml
# 3. Isaac Sim style: bash ~/isaacsim/python.sh ~/Hunav_isaac_wrapper/scripts/main.py
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Source ROS 2 and workspace if needed
if [ -z "$ROS_DISTRO" ]; then
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
    elif [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
    fi
fi
if [ -f "/workspace/hunav_ws/install/setup.bash" ]; then
    source /workspace/hunav_ws/install/setup.bash
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/src/scripts/main.py"

# Check if main script exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo -e "${YELLOW}Warning: main.py not found at $MAIN_SCRIPT${NC}"
    echo "Trying package installation location..."
    
    # Try to find via ROS2 package
    if command -v ros2 &> /dev/null; then
        PACKAGE_PATH=$(ros2 pkg prefix hunav_isaac_wrapper 2>/dev/null || echo "")
        if [ -n "$PACKAGE_PATH" ]; then
            MAIN_SCRIPT="$PACKAGE_PATH/lib/hunav_isaac_wrapper/main.py"
        fi
    fi
    
    if [ ! -f "$MAIN_SCRIPT" ]; then
        echo "Error: Cannot find main.py script"
        exit 1
    fi
fi

echo -e "${GREEN}HuNav Isaac Wrapper Launcher${NC}"
echo "Using script: $MAIN_SCRIPT"

# Check if Isaac Sim python is available.
# Order matches find_isaac_python() in src/hunav_isaac_wrapper/ros_launcher.py:
# container install first, then workstation install, then the AppImage layout.
ISAAC_PYTHON=""
if [ -f "/isaac-sim/python.sh" ]; then
    ISAAC_PYTHON="bash /isaac-sim/python.sh"
    echo "Using Isaac Sim python: $ISAAC_PYTHON"
elif [ -f "$HOME/isaacsim/python.sh" ]; then
    ISAAC_PYTHON="bash $HOME/isaacsim/python.sh"
    echo "Using Isaac Sim python: $ISAAC_PYTHON"
else
    echo "Isaac Sim python not found in standard locations. Searching for AppImage layout..."
    # ISAAC_SIM_PATH=$(ls "$HOME/.local/share/ov/pkg/isaac_sim-"*/python.sh 2>/dev/null | head -1)
    # if [ -n "$ISAAC_SIM_PATH" ] && [ -f "$ISAAC_SIM_PATH" ]; then
    #     ISAAC_PYTHON="bash $ISAAC_SIM_PATH"
    #     echo "Using Isaac Sim python: $ISAAC_PYTHON"
    # elif command -v isaacsim &> /dev/null; then
    #     ISAAC_PYTHON="isaacsim"
    #     echo "Using Isaac Sim python: $ISAAC_PYTHON"
    # fi
fi

if [ -z "$ISAAC_PYTHON" ]; then
    echo -e "${YELLOW}Error: Isaac Sim python not found.${NC}"
    echo "Searched:"
    echo "  /isaac-sim/python.sh"
    echo "  $HOME/isaacsim/python.sh"
    echo "  $HOME/.local/share/ov/pkg/isaac_sim-*/python.sh"
    echo "  isaacsim on PATH"
    echo ""
    echo "The simulation cannot run under the system python3 -- 'from isaacsim import"
    echo "SimulationApp' is only importable from Isaac Sim's own interpreter."
    exit 1
fi

# Check for environment-based NavMesh helper flag (HUNAV_NAVMESH_HELPER=1 or NAVMESH_HELPER=1)
if [ "${HUNAV_NAVMESH_HELPER:-0}" = "1" ] || [ "${NAVMESH_HELPER:-0}" = "1" ] || [ "${HUNAV_NAVMESH_HELPER:-}" = "true" ] || [ "${NAVMESH_HELPER:-}" = "true" ]; then
    has_navmesh_flag=false
    for arg in "$@"; do
        if [ "$arg" = "--navmesh-helper" ]; then
            has_navmesh_flag=true
            break
        fi
    done
    if [ "$has_navmesh_flag" = false ]; then
        echo -e "${GREEN}Enabling NavMesh helper from environment (HUNAV_NAVMESH_HELPER=1)${NC}"
        set -- "$@" --navmesh-helper
    fi
fi

# Parse arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "HuNav Isaac Wrapper Launcher"
    echo ""
    echo "Usage:"
    echo "  $0                                      # Interactive mode"
    echo "  $0 [scenario.yaml] [options...]         # Launch with specific scenario"
    echo "  $0 [args...]                            # Pass arguments to main.py"
    echo ""
    echo "Options forwarded to main.py:"
    echo "  --navmesh-helper                         # Auto-visualize NavMesh (/World/navmeshmesh) & open UI window"
    echo "  --flat-ground                            # Flatten terrain collision for even ground"
    echo "  --terrain-follow                         # Raycast terrain elevation dynamically for agents"
    echo "  --step-height M                          # Max step height in metres (default: 0.25)"
    echo "  --batch, -b                              # Batch mode (skip interactive prompts)"
    echo ""
    echo "Environment variables:"
    echo "  HUNAV_NAVMESH_HELPER=1                   # Same as --navmesh-helper"
    echo "  LIVESTREAM=1                             # Serve viewport via WebRTC on port 49100"
    echo "  LIVESTREAM_PUBLIC_IP=<IP>                # Announce public IP to WebRTC clients"
    echo "  HUNAV_CROWD_PROFILING=1                  # Print per-frame crowd update timings"
    echo "  HUNAV_ANIM_DEBUG=1                       # Print debug logs for agent animations"
    echo ""
    echo "Examples:"
    echo "  $0                                      # Show interactive menu"
    echo "  $0 warehouse_agents.yaml                # Launch warehouse scenario"
    echo "  $0 warehouse_agents.yaml --navmesh-helper # Launch with NavMesh visualizer and UI"
    echo "  $0 --config brownstone_agents.yaml --flat-ground --navmesh-helper --batch"
    echo "  HUNAV_NAVMESH_HELPER=1 $0 --config brownstone_agents.yaml --flat-ground --batch"
    echo ""
    exit 0
elif [ $# -eq 0 ]; then
    # Interactive mode
    echo -e "${GREEN}Launching in interactive mode...${NC}"
    $ISAAC_PYTHON "$MAIN_SCRIPT"
elif [ $# -eq 1 ] && [[ ! "$1" =~ ^-- ]]; then
    # Single scenario specified (doesn't start with --)
    echo -e "${GREEN}Launching with scenario: $1${NC}"
    $ISAAC_PYTHON "$MAIN_SCRIPT" --config "$1" --batch
else
    # Multiple arguments or arguments that start with --
    if [ $# -ge 2 ] && [[ ! "$1" =~ ^-- ]]; then
        # First argument is a scenario file, convert it to --config format
        SCENARIO="$1"
        shift  # Remove first argument
        echo -e "${GREEN}Launching with scenario: $SCENARIO and additional arguments: $@${NC}"
        $ISAAC_PYTHON "$MAIN_SCRIPT" --config "$SCENARIO" "$@"
    else
        # All arguments start with -- or it's a single -- argument
        echo -e "${GREEN}Launching with arguments: $@${NC}"
        $ISAAC_PYTHON "$MAIN_SCRIPT" "$@"
    fi
fi
