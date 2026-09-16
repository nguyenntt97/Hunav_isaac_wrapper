#!/usr/bin/env python3
"""
world_builder.py

A simple module that loads a USD stage (map) from disk, replacing the current stage.
"""

import os
import omni

class WorldBuilder:
    """
    Loads an entire USD stage from disk.
    
    """
    def __init__(self, base_path):
        self.base_path = base_path
        self.usd_context = omni.usd.get_context()

    def load_map(self, map_name: str):
        """
        Looks for `map_name.usd` inside the 'worlds' folder under base_path and opens it.

        Returns True if the stage was opened. A missing map is reported but not
        raised, so callers that post-process the stage must check the result.
        """
        map_path = os.path.join(self.base_path, "worlds", f"{map_name}.usd")
        if os.path.exists(map_path):
            self.usd_context.open_stage(map_path)
            print(f"Map '{map_name}' loaded from: {map_path}")
            return True
        print(f"[WorldBuilder] Error: map '{map_name}' not found at {map_path}")
        return False

    def get_stage(self):
        """The currently open USD stage, or None if nothing has been opened."""
        return self.usd_context.get_stage()
